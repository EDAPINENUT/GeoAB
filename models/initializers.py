import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_scatter import scatter_sum

import numpy as np
from data.pdb_utils import VOCAB
# from modules import RelationEGNN
import math
from .featurizers import *
from .nerf import nerf_build_batch
from .vonmises import VonMisesMix
from .modules import RelationEGNN, AtomEncoder

DIHEDRAL_TYPES = ['N-CA-C-N', 'CA-C-N-CA', 'C-N-CA-C', 'O=C-N-CA']
ANGLE_TYPES = []
BOND_TYPES = []
C_N_LENGTH = 1.34


class RotationInitializer(nn.Module):
    def __init__(self, 
                 embed_size, 
                 hidden_size, 
                 n_channel, 
                 n_layers, 
                 dropout, 
                 cdr_type, 
                 alpha, 
                 n_iter, 
                 node_feats_mode, 
                 edge_feats_mode, 
                 n_mixture=4,
                 n_layers_update=4,
                 noise_scale=0.0):
        super().__init__()
        self.alpha = alpha
        self.n_iter = n_iter
        self.cdr_type = cdr_type
        self.n_mixture = n_mixture
        self.noise_scale = noise_scale
        node_feats_dim = int(node_feats_mode[0]) * 16 + int(node_feats_mode[1]) * 48 + int(node_feats_mode[2]) * 12 + int(node_feats_mode[3]) * 9
        edge_feats_dim = int(edge_feats_mode[0]) * 16 + int(edge_feats_mode[1]) * 64 + int(edge_feats_mode[2]) * 4 + int(edge_feats_mode[3]) * 12 + 8

        self.num_aa_type = len(VOCAB)
        self.mask_token_id = VOCAB.get_unk_idx()
        self.protein_feature = ProteinFeaturizer(edge_feats_mode)
        self.fullatom_feature = FullAtomFeaturizer()

        self.aa_embedding = nn.Embedding(self.num_aa_type, embed_size)
        self.gnn = RelationEGNN(embed_size, hidden_size, self.num_aa_type, n_channel, n_layers=n_layers, dropout=dropout, node_feats_dim=node_feats_dim, edge_feats_dim=edge_feats_dim)
        
        self.atom_gnn = AtomEncoder(hidden_size, n_layers_update, dropout=dropout)
        self.dihedral_heads = nn.ModuleList([LeakyMLP(hidden_size*4 + 2, hidden_size, num_layer=n_layers_update, dim_end=n_mixture*3) for i in range(4)])


    def seq_loss(self, _input, target):
        return F.cross_entropy(_input, target, reduction='none')

    def coord_loss(self, _input, target):
        return F.smooth_l1_loss(_input, target, reduction='sum')
    
        
    def infer_with_nerf(self, X, 
                        cdr_range, 
                        diheds, 
                        angles=None,
                        distances=None
                        ):

        dihed_cncac = diheds[2]
        dihed_ncacn = diheds[0]
        dihed_cacnca = diheds[1]
        dihed_ocnca = diheds[3]
        if len(dihed_cacnca.shape) == 1:
            dihed_cacnca = dihed_cacnca.unsqueeze(0)
            dihed_ncacn = dihed_ncacn.unsqueeze(0)
            dihed_cncac = dihed_cncac.unsqueeze(0)
            dihed_ocnca = dihed_ocnca.unsqueeze(0)

        if angles is not None:
            angle_cnca = angles[2]
            angle_ncac = angles[0]
            angle_cacn = angles[1]
            angle_ocn = angles[3]

            if len(angle_cnca.shape)==1:
                angle_cnca = angle_cnca.unsqueeze(0)
                angle_ncac = angle_ncac.unsqueeze(0)
                angle_cacn = angle_cacn.unsqueeze(0)
                angle_ocn = angle_ocn.unsqueeze(0)

        if distances is not None:
            distance_nca = distances[0]
            distance_cac = distances[1]
            distance_co = distances[2]
            distance_cn = distances[3]
            if len(distance_nca.shape)==1:
                distance_nca = distance_nca.unsqueeze(0)
                distance_cac = distance_cac.unsqueeze(0)
                distance_co = distance_co.unsqueeze(0)
                distance_cn = distance_cn.unsqueeze(0)
        
        offset = 0
        X_valid = X.clone()

        X = X.unsqueeze(0).repeat(dihed_cncac.shape[0], 1, 1, 1)
        for start, end in cdr_range:
            dihed_cncac_cdr = dihed_cncac[:, offset: offset + end + 1 - start]
            dihed_ncacn_cdr = dihed_ncacn[:, offset: offset + end + 1 - start]
            dihed_cacnca_cdr = dihed_cacnca[:, offset: offset + end + 1 - start]
            dihed_ocnca_cdr = dihed_ocnca[:, offset: offset + end + 1 - start]
            if angles is not None:
                angle_cnca_cdr = angle_cnca[:, offset: offset + end + 1 - start]
                angle_ncac_cdr = angle_ncac[:, offset: offset + end + 1 - start]
                angle_cacn_cdr = angle_cacn[:, offset: offset + end + 1 - start]
                angle_ocn_cdr = angle_ocn[:, offset: offset + end + 1 - start]
            else:
                angle_cnca_cdr, angle_ncac_cdr, angle_cacn_cdr, angle_ocn_cdr = None, None, None, None

            if distances is not None:
                distance_nca_cdr = distance_nca[:, offset: offset + end + 1 - start]
                distance_cac_cdr = distance_cac[:, offset: offset + end + 1 - start]
                distance_co_cdr = distance_co[:, offset: offset + end + 1 - start]
                distance_cn_cdr = distance_cn[:, offset: offset + end + 1 - start]
            else:
                distance_nca_cdr, distance_cac_cdr, distance_co_cdr, distance_cn_cdr = None, None, None, None

            
            offset += end + 1 - start

            coord_cdr = nerf_build_batch(phi=dihed_cncac_cdr, 
                                         psi=dihed_ncacn_cdr, 
                                         omega=dihed_cacnca_cdr,
                                        bond_angle_c_n_ca=angle_cnca_cdr,
                                        bond_angle_ca_c_n=angle_cacn_cdr,
                                        bond_angle_n_ca_c=angle_ncac_cdr,
                                        bond_len_c_n=distance_cn_cdr,
                                        bond_len_ca_c=distance_cac_cdr,
                                        bond_len_n_ca=distance_nca_cdr,
                                        dihed_o_c_n_ca=dihed_ocnca_cdr,
                                        bond_angle_o_c_n=angle_ocn_cdr,
                                        bond_len_o_c=distance_co_cdr,
                                        init_pos=X[0, start-1, :3],
                                        final_pos=X[0, end+1, :3],
                                        X_true = X[0, start-1:end+1, :3],)

            coord_cdr = coord_cdr.view(coord_cdr.shape[0],end-start+2, 4, -1)[:, 1:]

            # X[:, start:end + 1] = coord_cdr
            choosen_idx = torch.argmin((torch.norm(coord_cdr[:, -1, 2, :] - X[:, end+2,0,:], dim=-1) - C_N_LENGTH).abs())
            X_valid[start:end+1] = coord_cdr[choosen_idx]
        
        return X_valid
        

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
            X[start:end + 1] += torch.rand_like(mask_coords) * self.noise_scale
            cmask[start:end + 1, ...] = 1
        return X, S, cmask

    
    @torch.no_grad()
    def generate(self, X, S, L, offsets, greedy):
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)
         
        X_true, S_true = X.clone(), S.clone()

        # init mask
        X, S, cmask = self.init_mask(X, S, cdr_range)  # [n_all_node, n_channel, 3]
        mask = cmask[:, 0, 0].bool()  # [n_all_node]
        aa_cnt = mask.sum()

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        smask = special_mask.repeat(aa_cnt, 1).bool()
        H_0 = self.aa_embedding(S)  # [n_all_node, embed_size]

        with torch.no_grad():     
            edges_list, edge_feats_list, node_feats, segment_ids, segment_idx = self.protein_feature(X, S, offsets)
            (atoms, atom_edge_index, atom_edge_attr, 
                (intra_geom_idx_list, intra_geom_type_list),
                (intra_geom_cdr_idx_list, intra_geom_cdr_type_list)) = self.fullatom_feature(segment_ids, segment_idx, S, cdr_range)
            geom_inits = self.get_target_geom(X, intra_geom_cdr_idx_list, intra_geom_cdr_type_list)
            dihed_init = geom_inits[2]

        H, Z, aa_emb = self.gnn(H_0, X, edges_list, edge_feats_list, node_feats)
        atom_emb = self.atom_gnn(Z, S, aa_emb, 
                                bonds=intra_geom_idx_list[0], 
                                bond_types=intra_geom_type_list[0], 
                                angles=intra_geom_idx_list[1], 
                                angle_types=intra_geom_type_list[1], 
                                torsions=intra_geom_idx_list[2], 
                                torsion_types=intra_geom_type_list[2]
                                )
        pred_diheds = []
        for i in range(4):
            pred_dihed = self.dihedral_heads[i](
                torch.cat([
                    atom_emb[intra_geom_cdr_idx_list[2][:,0][intra_geom_cdr_type_list[2]==i]],
                    atom_emb[intra_geom_cdr_idx_list[2][:,1][intra_geom_cdr_type_list[2]==i]],
                    atom_emb[intra_geom_cdr_idx_list[2][:,2][intra_geom_cdr_type_list[2]==i]],
                    atom_emb[intra_geom_cdr_idx_list[2][:,3][intra_geom_cdr_type_list[2]==i]],
                    torch.cos(dihed_init[intra_geom_cdr_type_list[2]==i]).unsqueeze(-1),
                    torch.sin(dihed_init[intra_geom_cdr_type_list[2]==i]).unsqueeze(-1),
                ], dim=-1)
            )
            pred_diheds.append(pred_dihed)

        gen_diheds = self.von_mises_generate(pred_diheds)
        X_gen = self.infer_with_nerf(X_true, cdr_range, diheds=gen_diheds)
        return X_gen, X_true, cdr_range

    def von_mises_generate(self, preds, sample_num = 100):
        sample_angles = []
        
        for i, pred in enumerate(preds):
            loc_pred, conc_pred, logits = pred.split(self.n_mixture, dim=-1)
            distribution = VonMisesMix(loc_pred, conc_pred, logits)
            angles = distribution.sample(sample_num)
            sample_angles.append(angles)
        return sample_angles
    

    def infer(self, batch, device, greedy=True):
        X, S, L, offsets = batch['X'].to(device), batch['S'].to(device), batch['L'], batch['offsets'].to(device)
        X_pred, X_true, cdr_range = self.generate(X, S, L, offsets, greedy=greedy)

        cdr_range = cdr_range.tolist()
        X_pred, X_true = X_pred.cpu().numpy(), X_true.cpu().numpy()

        seq, x, x_true = [], [], []
        for start, end in cdr_range:
            end = end + 1
            seq.append(''.join([VOCAB.idx_to_symbol(S[i]) for i in range(start, end)]))
            x.append(X_pred[start:end])
            x_true.append(X_true[start:end])

        
        return seq, x, x_true, True


    def forward(self, X, S, L, offsets):
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
        X_true, S_true = X.clone(), S.clone()

        # init mask
        X, S, cmask = self.init_mask(X, S, cdr_range)  # [n_all_node, n_channel, 3]
        mask = cmask[:, 0, 0].bool()  # [n_all_node]
        aa_cnt = mask.sum()

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        smask = special_mask.repeat(aa_cnt, 1).bool()
        H_0 = self.aa_embedding(S)  # [n_all_node, embed_size]
        aa_embeddings = self.aa_embedding(torch.arange(self.num_aa_type, device=H_0.device))  # [vocab_size, embed_size]

        # initialization
        with torch.no_grad():     
            edges_list, edge_feats_list, node_feats, segment_ids, segment_idx = self.protein_feature(X, S, offsets)
            (atoms, atom_edge_index, atom_edge_attr, 
            (intra_geom_idx_list, intra_geom_type_list),
            (intra_geom_cdr_idx_list, intra_geom_cdr_type_list)) = self.fullatom_feature(segment_ids, segment_idx, S, cdr_range)
            geom_targets = self.get_target_geom(X_true, intra_geom_cdr_idx_list, intra_geom_cdr_type_list)
            geom_inits = self.get_target_geom(X, intra_geom_cdr_idx_list, intra_geom_cdr_type_list)
            dihed_target = geom_targets[2]
            dihed_init = geom_inits[2]

        H, Z, aa_emb = self.gnn(H_0, X, edges_list, edge_feats_list, node_feats)
        atom_emb = self.atom_gnn(Z, S, aa_emb, 
                                bonds=intra_geom_idx_list[0], 
                                bond_types=intra_geom_type_list[0], 
                                angles=intra_geom_idx_list[1], 
                                angle_types=intra_geom_type_list[1], 
                                torsions=intra_geom_idx_list[2], 
                                torsion_types=intra_geom_type_list[2]
                                )
        pred_diheds = []
        target_diheds = []
        for i in range(4):
            pred_dihed = self.dihedral_heads[i](
                torch.cat([
                    atom_emb[intra_geom_cdr_idx_list[2][:,0][intra_geom_cdr_type_list[2]==i]],
                    atom_emb[intra_geom_cdr_idx_list[2][:,1][intra_geom_cdr_type_list[2]==i]],
                    atom_emb[intra_geom_cdr_idx_list[2][:,2][intra_geom_cdr_type_list[2]==i]],
                    atom_emb[intra_geom_cdr_idx_list[2][:,3][intra_geom_cdr_type_list[2]==i]],
                    torch.cos(dihed_init[intra_geom_cdr_type_list[2]==i]).unsqueeze(-1),
                    torch.sin(dihed_init[intra_geom_cdr_type_list[2]==i]).unsqueeze(-1),
                ], dim=-1)
            )
            pred_diheds.append(pred_dihed)
            target_diheds.append(dihed_target[[intra_geom_cdr_type_list[2]==i]])

        loss = self.von_mises_loss(pred_diheds, target_diheds)
        return loss.mean(), [l.item() for l in loss]

       
    
    def von_mises_loss(self, preds, targets, loss_alphas = [0.4, 0.1, 0.4, 0.1]):
        type_losses = torch.zeros(len(preds))
        
        for i, (pred, target) in enumerate(zip(preds, targets)):
            loc_pred, conc_pred, logits = pred.split(self.n_mixture, dim=-1)
            distribution = VonMisesMix(loc_pred, conc_pred, logits)
            type_losses[i] = -1.0 * loss_alphas[i] * distribution.log_prob(target.unsqueeze(-1)).mean() # negative log-likelihood
        return type_losses

    def get_target_geom(self, X, geom_idx_list, geom_type_list):
        X = X.reshape(-1, 3)
        geom_targets = []
        for geom_idx, geom_type in zip(geom_idx_list, geom_type_list):
            if geom_idx.shape[1] == 4:
                r1 = X[geom_idx[:,1]] - X[geom_idx[:,0]]
                r2 = X[geom_idx[:,2]] - X[geom_idx[:,1]]
                r3 = X[geom_idx[:,3]] - X[geom_idx[:,2]]
                dihed = cal_dihedral(r1, r2, r3)
                geom_targets.append(dihed)
            elif geom_idx.shape[1] == 3:
                r1 = F.normalize(X[geom_idx[:,0]] - X[geom_idx[:,1]], dim=-1)
                r2 = F.normalize(X[geom_idx[:,2]] - X[geom_idx[:,1]], dim=-1)
                angles = torch.acos((r1 * r2).sum(dim=-1))
                geom_targets.append(angles)
            elif geom_idx.shape[1] == 2:
                distance = (X[geom_idx[:,0]] - X[geom_idx[:,1]]).norm(dim=-1)
                geom_targets.append(distance)
        return geom_targets



def cal_dihedral(r1, r2, r3):
    n_1 = torch.cross(r1, r2, dim=-1)
    n_2 = torch.cross(r2, r3, dim=-1)
    r2_norm = torch.linalg.vector_norm(r2, dim=-1, keepdim=True)
    r1_r2 = r1 * r2_norm
    D = torch.atan2(
            (r1_r2 * n_2).sum(dim=-1),
            (n_1 * n_2).sum(dim=-1)
        )

    return D


class LeakyMLP(nn.Module):
    """A fairly generic MLP
    
    Parameters
    ----------
    dim_start : int
        Dimension of input
    dim_hidden : int
        Dimension of hidden layers
    dim_end : int, default 1
        Dimension of output
    num_layer : int, default 0
        Number of hidden layers
    leaky : float, default 0.001
        Value of leakiness given to the LeakyReLU activation layers
    """
    def __init__(self, dim_start: int, dim_hidden: int, dim_end: int=1, num_layer: int=0, leaky: float=0.001):
        super().__init__()

        modules = [nn.Linear(dim_start, dim_hidden), nn.LeakyReLU(leaky)]
        for _ in range(num_layer):
            modules.append(nn.Linear(dim_hidden, dim_hidden))
            modules.append(nn.LayerNorm(dim_hidden))
            modules.append(nn.LeakyReLU(leaky))
        modules.append(nn.Linear(dim_hidden, dim_end))

        self.decode = nn.ModuleList(modules)

    def forward(self, x):
        for d in self.decode:
            x = d(x)
        return x



class TrigoEmbedding(nn.Module):

    def __init__(self, pos_min=0.0, pos_max=1.0, dim_embedding=64, encode_angle=True, fourier=True, half_encode=False):
        super().__init__()
        self.pos_min = pos_min if not encode_angle else pos_min * math.pi
        self.pos_max = pos_max if not encode_angle else pos_max * math.pi
        self.dim_embedding = dim_embedding
        offset = torch.linspace(pos_min, pos_max, self.dim_embedding // 2 + 1)[1:]    # 1 overflow flag
        self.fourier = fourier
        self.half_encode = half_encode
        if self.fourier: 
            self.fourier_coeff = nn.Linear(dim_embedding, dim_embedding)
        self.register_buffer('offset', offset)

    @property
    def out_channels(self):
        return self.dim_embedding 

    def forward(self, pos, dim=-1, normalize=False):

        assert pos.size()[dim] == 1
        offset_shape = [1] * len(pos.size())
        offset_shape[dim] = -1

        y_ = pos * self.offset.view(*offset_shape)  # (N, *, dim_embedding-1, *)
        y_sin = torch.sin(y_)  # (N, *, dim_embedding-1, *)
        y_cos = torch.cos(y_)  # (N, *, dim_embedding, *)
        y = torch.cat([y_sin, y_cos], dim=-1) if not self.half_encode else torch.cat([y_cos,y_cos], dim=-1)
        if normalize:
            y = y / y.sum(dim=dim, keepdim=True)
        if self.fourier:
            y = self.fourier_coeff(y)
        return y
    
class Smearing(nn.Module):

    def __init__(self, dist_min=0.0, dist_max=20.0, dim_embedding=64, use_onehot=False):
        super().__init__()
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.dim_embedding = dim_embedding
        self.use_onehot = use_onehot

        if use_onehot:
            offset = torch.linspace(dist_min, dist_max, self.dim_embedding)
        else:
            offset = torch.linspace(dist_min, dist_max, self.dim_embedding-1)    # 1 overflow flag
            self.coeff = -0.5 / ((offset[1] - offset[0]) * 1.0).item() ** 2  # `*0.2`: makes it not too blurred
        self.register_buffer('offset', offset)

    @property
    def out_channels(self):
        return self.dim_embedding 

    def forward(self, dist, dim=-1, normalize=True):

        assert dist.size()[dim] == 1
        offset_shape = [1] * len(dist.size())
        offset_shape[dim] = -1

        if self.use_onehot:
            diff = torch.abs(dist - self.offset.view(*offset_shape))  # (N, *, dim_embedding, *)
            bin_idx = torch.argmin(diff, dim=dim, keepdim=True)  # (N, *, 1, *)
            y = torch.zeros_like(diff).scatter_(dim=dim, index=bin_idx, value=1.0)
        else:
            overflow_symb = (dist >= self.dist_max).float()  # (N, *, 1, *)
            y = dist - self.offset.view(*offset_shape)  # (N, *, dim_embedding-1, *)
            y = torch.exp(self.coeff * torch.pow(y, 2))  # (N, *, dim_embedding-1, *)
            y = torch.cat([y, overflow_symb], dim=dim)  # (N, *, dim_embedding, *)
            if normalize:
                y = y / y.sum(dim=dim, keepdim=True)

        return y

def inflate_segment_id(segment_id, data):
    inflate_size = data.size()[1:]
    length = len(inflate_size)
    unsqueeze_dims = tuple([1 for i in range(length)])
    segment_id = segment_id.view(-1, *unsqueeze_dims).expand(-1, *inflate_size)
    return segment_id

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, *data.size()[1:])
    segment_ids = inflate_segment_id(segment_ids, data)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
