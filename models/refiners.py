import torch
from torch import nn
import torch.nn.functional as F
from .featurizers import *
from data.pdb_utils import VOCAB
from .modules import RelationEGNN, BondLengthMechanics, BondAngleMechanics, TorsionAngleMechanics, TransformerEncoder
from data.esm_utils import ESM_DIM

class GeoRefiner(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, n_layers, 
                 dropout, cdr_type, alpha, node_feats_mode, edge_feats_mode, 
                 interface_only, beta=lambda x: 0.1, n_iter=1, n_layers_update=2, 
                 local_update=True, use_esm=True, initializer=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.cdr_type = cdr_type
        self.use_esm = use_esm
        
        esm_dim = ESM_DIM if self.use_esm else 0

        node_feats_dim = int(node_feats_mode[0]) * 16 + int(node_feats_mode[1]) * 48 + int(node_feats_mode[2]) * 12 + int(node_feats_mode[3]) * 9 + esm_dim
        edge_feats_dim = int(edge_feats_mode[0]) * 16 + int(edge_feats_mode[1]) * 64 + int(edge_feats_mode[2]) * 4 + int(edge_feats_mode[3]) * 12 + 8

        self.num_aa_type = len(VOCAB)
        self.mask_token_id = VOCAB.get_unk_idx()
        
        self.aa_embedding = nn.Embedding(self.num_aa_type, embed_size)
        self.gnn = RelationEGNN(embed_size, hidden_size, self.num_aa_type, n_channel, n_layers=n_layers, dropout=dropout, node_feats_dim=node_feats_dim, edge_feats_dim=edge_feats_dim)
        self.mechanics_update = Mechanics(hidden_size, n_layers_update, dropout=dropout) if local_update else None 
        self.protein_feature = ProteinFeaturizer(interface_only, use_esm=use_esm)
        self.fullatom_feature = FullAtomFeaturizer()
        self.initializer = initializer

    def seq_loss(self, _input, target):
        return F.cross_entropy(_input, target, reduction='none')

    def coord_loss(self, _input, target):
        return F.smooth_l1_loss(_input, target, reduction='sum')
    
    def geom_loss(self, geom_true, geom_pred):
        distance_loss = F.smooth_l1_loss(geom_pred[0], geom_true[0])
        # distance_loss = torch.zeros_like(geom_pred[0])
        angle_loss = F.smooth_l1_loss(geom_pred[1] , geom_true[1])
        dihed_loss = torch.zeros_like(geom_pred[2])
        # angle_loss = torch.zeros_like(geom_pred[1])
        # dihed_loss = -torch.cos(geom_pred[2] - geom_true[2])
        loss_list = [distance_loss.mean(), angle_loss.mean(), dihed_loss.mean()]
        loss_geom = sum(loss_list)
        return loss_geom, loss_list

    def init_mask(self, X, S, cdr_range):
        '''
        set coordinates of masks following a unified distribution
        between the two ends
        '''
        X, S, cmask = X.clone(), S.clone(), torch.zeros_like(X, device=X.device)
        n_channel, n_dim = X.shape[1:]
        for start, end in cdr_range:
            S[start:end + 1] = self.mask_token_id
            l_coord, r_coord = X[start - 1], X[end + 1]  # [n_channel, 3]
            n_span = end - start + 2
            coord_offsets = (r_coord - l_coord).unsqueeze(0).expand(n_span - 1, n_channel, n_dim)  # [n_mask, n_channel, 3]
            coord_offsets = torch.cumsum(coord_offsets, dim=0)
            mask_coords = l_coord + coord_offsets / n_span
            X[start:end + 1] = mask_coords
            cmask[start:end + 1, ...] = 1
        return X, S, cmask
    
    def init_with_pretrained(self, X, S, L, offsets, cdr_range, greedy=False):
        X, S, cmask = self.init_mask(X, S, cdr_range)  # [n_all_node, n_channel, 3]
        X_gen, _, cdr_range = self.initializer.generate(X, S, L, offsets, greedy)
        return X_gen, S, cmask
    
    def rmsd(self, X, true_X, mask):
        '''
        :param X: [n_all_node, n_channel, 3]
        :param true_X: [n_all_node, n_channel, 3]
        :param mask: [n_all_node]
        '''
        X, true_X = X[mask], true_X[mask]
        return torch.sqrt(torch.sum((X - true_X)**2, dim=-1)).mean()

    def forward(self, X, S, L, offsets, ESMs=None, iter_num=1):
        '''
        :param X: [n_all_node, n_channel, 3]
        :param S: [n_all_node]
        :param L: list of cdr types
        :param offsets: [batch_size + 1]
        '''
        # prepare inputs
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)

        # save ground truth
        true_X, true_S = X.clone(), S.clone()

        # init mask
        X, S, cmask = self.init_mask(X, S, cdr_range) if self.initializer is None else self.init_with_pretrained(X, S, L, offsets, cdr_range) # [n_all_node, n_channel, 3]
        mask = cmask[:, 0, 0].bool()  # [n_all_node]
        aa_cnt = mask.sum()

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        smask = special_mask.repeat(aa_cnt, 1).bool()
        H_0 = self.aa_embedding(S)  # [n_all_node, embed_size]
        aa_embeddings = self.aa_embedding(torch.arange(self.num_aa_type, device=H_0.device))  # [vocab_size, embed_size]

        snll, closs = 0, 0

        for r in range(self.n_iter):
            with torch.no_grad():
                edges_list, edge_feats_list, node_feats, segment_ids, segment_idx = self.protein_feature(X, S, offsets, ESMs=ESMs)
                (atoms, atom_edge_index, atom_edge_attr, 
                 (intra_geom_idx_list, intra_geom_type_list),
                 (intra_geom_cdr_idx_list, intra_geom_cdr_type_list)) = self.fullatom_feature(segment_ids, segment_idx, S, cdr_range)
            H, Z, aa_emb = self.gnn(H_0, X, edges_list, edge_feats_list, node_feats)

            if self.mechanics_update and r == (self.n_iter - 1):
                Z_refined = Z.detach().clone()
                aa_emb_refined = aa_emb.detach().clone()
                Z_local = self.mechanics_update(Z_refined, atoms, aa_emb_refined, 
                                                atom_edge_index, atom_edge_attr,
                                                bonds=intra_geom_idx_list[0], 
                                                bond_types=intra_geom_type_list[0], 
                                                angles=intra_geom_idx_list[1], 
                                                angle_types=intra_geom_type_list[1], 
                                                torsions=intra_geom_idx_list[2], 
                                                torsion_types=intra_geom_type_list[2]
                                                ).reshape_as(Z)

            # refinement
            X = X.clone()
            X[mask] = Z[mask]

            H_0 = H_0.clone()
            logits = H[mask]
            seq_prob = torch.softmax(logits.masked_fill(smask, float('-inf')), dim=-1)  # [aa_cnt, vocab_size]
            H_0[mask] = seq_prob.mm(aa_embeddings)  # smooth embedding
            
            r_snll = torch.sum(self.seq_loss(logits, true_S[mask])) / aa_cnt
            snll += r_snll / self.n_iter
    
        closs = self.coord_loss(Z[mask], true_X[mask]) / aa_cnt

        if self.mechanics_update:
            geom_pred = self.fullatom_feature.cal_geom_val(Z_local, intra_geom_cdr_idx_list, intra_geom_cdr_type_list)
            geom_true = self.fullatom_feature.cal_geom_val(true_X, intra_geom_cdr_idx_list, intra_geom_cdr_type_list)
            closs_local = self.coord_loss(Z_local[mask], true_X[mask]) / aa_cnt
            geom_loss, geom_loss_list = self.geom_loss(geom_true, geom_pred)
        else:
            closs_local = torch.zeros(1, device=X.device)
            geom_loss = torch.zeros(1, device=X.device)
            geom_loss_list = torch.zeros(3, device=X.device)

        loss = snll + self.alpha * closs + self.alpha * closs_local + self.beta(iter_num) * geom_loss

        # rmsd = self.rmsd(Z, true_X, mask)
        rmsd = self.rmsd(Z_local, true_X, mask) if self.mechanics_update else self.rmsd(Z, true_X, mask)
        
        return loss, r_snll, closs, closs_local, geom_loss, rmsd  # only return the last snll

    def generate(self, X, S, L, offsets, ESMs=None, greedy=True):
        '''
        :param X: [n_all_node, n_channel, 3]
        :param S: [n_all_node]
        :param L: list of cdr types
        :param offsets: [batch_size + 1]
        '''
        # prepare inputs
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)

        # save ground truth
        true_X, true_S = X.clone(), S.clone()

        # init mask
        X, S, cmask = self.init_mask(X, S, cdr_range)  if self.initializer is None else self.init_with_pretrained(X, S, L, offsets, cdr_range)
        mask = cmask[:, 0, 0].bool()  # [n_all_node]
        aa_cnt = mask.sum()

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        smask = special_mask.repeat(aa_cnt, 1).bool()
        H_0 = self.aa_embedding(S)  # [n_all_node, embed_size]
        aa_embeddings = self.aa_embedding(torch.arange(self.num_aa_type, device=H_0.device))  # [vocab_size, embed_size]
        
        for r in range(self.n_iter):
            with torch.no_grad():
                edges_list, edge_feats_list, node_feats, segment_ids, segment_idx = self.protein_feature(X, S, offsets, ESMs=ESMs)
                (atoms, atom_edge_index, atom_edge_attr, 
                 (intra_geom_idx_list, intra_geom_type_list),
                 (intra_geom_cdr_idx_list, intra_geom_cdr_type_list)) = self.fullatom_feature(segment_ids, segment_idx, S, cdr_range)
            H, Z, aa_emb = self.gnn(H_0, X, edges_list, edge_feats_list, node_feats)

            if self.mechanics_update and r == (self.n_iter - 1):
                Z_refined = Z.detach().clone()
                aa_emb_refined = aa_emb.detach().clone()
                Z_local = self.mechanics_update(Z_refined, atoms, aa_emb_refined, 
                                                atom_edge_index, atom_edge_attr,
                                                bonds=intra_geom_idx_list[0], 
                                                bond_types=intra_geom_type_list[0], 
                                                angles=intra_geom_idx_list[1], 
                                                angle_types=intra_geom_type_list[1], 
                                                torsions=intra_geom_idx_list[2], 
                                                torsion_types=intra_geom_type_list[2]
                                                ).reshape_as(Z)

        # refinement
        X = X.clone()
        X[mask] = Z_local[mask] if self.mechanics_update else Z[mask]

        H_0 = H_0.clone()
        seq_prob = torch.softmax(H[mask].masked_fill(smask, float('-inf')), dim=-1)  # [aa_cnt, vocab_size]
        H_0[mask] = seq_prob.mm(aa_embeddings)  # smooth embedding

        logits = H[mask]  # [aa_cnt, vocab_size]
        logits = logits.masked_fill(smask, float('-inf'))  # mask special tokens

        if greedy:
            S[mask] = torch.argmax(logits, dim=-1)  # [n]
        else:
            prob = F.softmax(logits, dim=-1)
            S[mask] = torch.multinomial(prob, num_samples=1).squeeze()
        snll_all = self.seq_loss(logits, S[mask])

        return snll_all, S, X, true_X, cdr_range

    def infer(self, batch, device, greedy=True):
        X, S, L, offsets, ESMs = batch['X'].to(device), batch['S'].to(device), batch['L'], batch['offsets'].to(device), batch['ESMs'].to(device)
        snll_all, pred_S, pred_X, true_X, cdr_range = self.generate(X, S, L, offsets, ESMs, greedy=greedy)

        pred_S, cdr_range = pred_S.tolist(), cdr_range.tolist()
        pred_X, true_X = pred_X.detach().cpu().numpy(), true_X.detach().cpu().numpy()

        # seqs, x, true_x
        seq, x, true_x = [], [], []
        for start, end in cdr_range:
            end = end + 1
            seq.append(''.join([VOCAB.idx_to_symbol(pred_S[i]) for i in range(start, end)]))
            x.append(pred_X[start:end])
            true_x.append(true_X[start:end])

        # ppl
        ppl = [0 for _ in range(len(cdr_range))]
        lens = [0 for _ in ppl]
        offset = 0

        for i, (start, end) in enumerate(cdr_range):
            length = end - start + 1
            for t in range(length):
                ppl[i] += snll_all[t + offset]
            offset += length
            lens[i] = length

        ppl = [p / n for p, n in zip(ppl, lens)]
        ppl = torch.exp(torch.tensor(ppl, device=device)).tolist()

        meta_x_true = []
        meta_x_pred = []
        meta_s_true = []
        meta_s_pred = []
        meta_l = []
        for k , (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            meta_x_true.append(true_X[start:end])
            meta_x_pred.append(pred_X[start:end])
            meta_s_true.append(S[start:end])
            meta_s_pred.append(pred_S[start:end])
            meta_l.append(L[k])
        
        meta_data = {'x_true': meta_x_true, 'x_pred': meta_x_pred, 's_true': meta_s_true, 's_pred': meta_s_pred, 'l': meta_l}
        return ppl, seq, x, true_x, True, meta_data
    
    

class Mechanics(nn.Module):
    def __init__(self, dim_embed, num_layer, dropout=0.1, dim_ffn=None, num_head=8, num_layer_mlp=2) -> None:
        super().__init__()
        self.dim_embed = dim_embed
        self.atom_embed = nn.Embedding(4, self.dim_embed)
        self.edge_embed = nn.Embedding(20, self.dim_embed)
        dim_ffn = 2*self.dim_embed if dim_ffn is None else dim_ffn

        self.num_layer = num_layer
        self.bond_length_mechanics = BondLengthMechanics(dim_embed, num_layer=num_layer_mlp)
        self.bond_angle_mechanics = BondAngleMechanics(dim_embed, num_layer=num_layer_mlp)
        self.torsion_angle_mechanics = TorsionAngleMechanics(dim_embed, num_layer=num_layer_mlp)
        self.gnn_atom = TransformerEncoder(
            node_input_dim=dim_embed,
            dim_embed=dim_embed,
            dim_ffn=dim_ffn,
            num_head=num_head,
            num_layer=num_layer,
            edge_input_dim=dim_embed,
            dropout=dropout,
            bi=True
        )

    
    def forward(self, pos, atoms, aa_embedding, edge_index, edge_type,
                bonds, bond_types, angles, angle_types, torsions, torsion_types) :
        atoms = atoms.reshape(aa_embedding.shape[0], 4)
        pos = pos.view(atoms.shape[0]*atoms.shape[1], -1)

        node_embedding = (self.atom_embed(atoms) + aa_embedding.unsqueeze(1)).view(atoms.shape[0]*atoms.shape[1], -1)
        edge_embedding = self.edge_embed(edge_type)
        node_embedding, _ = self.gnn_atom(node_embedding, edge_embedding, edge_index)
        
        pos_update = pos.clone()
        pos_update = self.bond_length_mechanics(pos, bonds, node_embedding, pos_update, bond_types)
        pos_update = self.bond_angle_mechanics(pos, angles, node_embedding, pos_update, angle_types)
        pos_update = self.torsion_angle_mechanics(pos, torsions, node_embedding, pos_update, torsion_types)
        
        return pos_update
    
