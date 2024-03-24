import torch 
from data.pdb_utils import VOCAB
import numpy as np
from torch_scatter import scatter_sum
from torch import nn
import torch.nn.functional as F

class PosEmbedding(nn.Module):

    def __init__(self, num_embeddings):
        super(PosEmbedding, self).__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx):
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        ).cuda()
        angles = E_idx.unsqueeze(-1) * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E
    
def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res

class FullAtomFeaturizer(nn.Module):
    def __init__(self, stable_angle=True, BOH_idx=2) -> None:
        super().__init__()
        self.BOH_idx = BOH_idx
        self.atom_idx = {'N': 0, 'CA': 1, 'C':2, 'O':3}
        self.bond_idx = {'NCA': 0, 'CAC': 1, 'CO': 2, 'CN': 3}
        self.angle_idx = {'NCAC': 0, 'CACN': 1, 'CNCA': 2, 'OCN': 3}
        self.dihed_idx = {'NCACN': 0, 'CACNCA': 1, 'CNCAC': 2, 'OCNCA': 3}
        self.stable_angle = stable_angle

    @torch.no_grad()
    def construct_bond_relation(self, index_group_atoms, mask_heavy_chains, mask_bridges, seq_len):
        nca_bond = index_group_atoms[:, [self.atom_idx['N'],self.atom_idx['CA']]][mask_heavy_chains]
        cac_bond = index_group_atoms[:, [self.atom_idx['CA'],self.atom_idx['C']]][mask_heavy_chains]
        co_bond = index_group_atoms[:, [self.atom_idx['C'],self.atom_idx['O']]][mask_heavy_chains]
        cn_bond = torch.stack(
            [index_group_atoms[:-1, self.atom_idx['C']], index_group_atoms[1:, self.atom_idx['N']]], 
            dim=-1)[mask_bridges]
        
        nca_bond_res_idx = torch.arange(seq_len).to(index_group_atoms)[mask_heavy_chains]
        cac_bond_res_idx = torch.arange(seq_len).to(index_group_atoms)[mask_heavy_chains]
        co_bond_res_idx = torch.arange(seq_len).to(index_group_atoms)[mask_heavy_chains]
        cn_bond_res_idx = torch.arange(seq_len).to(index_group_atoms)[1:][mask_bridges]
        
        bonds = torch.cat([nca_bond, cac_bond, co_bond, cn_bond], dim=0)
        bonds_res_idx = torch.cat([nca_bond_res_idx, 
                                   cac_bond_res_idx, 
                                   co_bond_res_idx, 
                                   cn_bond_res_idx], dim=0)
        bonds_type = torch.cat([torch.ones(nca_bond.shape[0]) * self.bond_idx['NCA'],
                                torch.ones(cac_bond.shape[0]) * self.bond_idx['CAC'],
                                torch.ones(co_bond.shape[0]) * self.bond_idx['CO'],
                                torch.ones(cn_bond.shape[0]) * self.bond_idx['CN']]).to(index_group_atoms)
        return bonds, bonds_type, bonds_res_idx
    
    @torch.no_grad()
    def construct_angle_relation(self, index_group_atoms, mask_heavy_chains, mask_bridges, seq_len):
        ncac_angle = index_group_atoms[:, [self.atom_idx['N'],self.atom_idx['CA'],self.atom_idx['C']]][mask_heavy_chains]
        cacn_angle = torch.cat(
            [index_group_atoms[:-1, [self.atom_idx['CA'],self.atom_idx['C']]], index_group_atoms[1:, [self.atom_idx['N']]]], 
            dim=-1)[mask_bridges]
        cnca_angle = torch.cat(
            [index_group_atoms[:-1, [self.atom_idx['C']]], index_group_atoms[1:, [self.atom_idx['N'],self.atom_idx['CA']]]], 
            dim=-1)[mask_bridges]
        ocn_angle = torch.cat(
            [index_group_atoms[:-1, [self.atom_idx['O'],self.atom_idx['C']]], index_group_atoms[1:, [self.atom_idx['N']]]], 
            dim=-1)[mask_bridges]
        ncac_angle_res_idx = torch.arange(seq_len).to(index_group_atoms)[mask_heavy_chains]
        cacn_angle_res_idx = torch.arange(seq_len).to(index_group_atoms)[1:][mask_bridges]
        cnca_angle_res_idx = torch.arange(seq_len).to(index_group_atoms)[1:][mask_bridges]
        ocn_angle_res_idx = torch.arange(seq_len).to(index_group_atoms)[1:][mask_bridges]
        
        angles = torch.cat([ncac_angle, cacn_angle, cnca_angle, ocn_angle], dim=0)
        angles_res_idx = torch.cat([ncac_angle_res_idx, 
                                   cacn_angle_res_idx, 
                                   cnca_angle_res_idx, 
                                   ocn_angle_res_idx], dim=0)
        angles_type = torch.cat([torch.ones(ncac_angle.shape[0]) * self.angle_idx['NCAC'],
                                torch.ones(cacn_angle.shape[0]) * self.angle_idx['CACN'],
                                torch.ones(cnca_angle.shape[0]) * self.angle_idx['CNCA'],
                                torch.ones(ocn_angle.shape[0]) * self.angle_idx['OCN']]).to(index_group_atoms)
        
        return angles, angles_type, angles_res_idx

    @torch.no_grad()
    def construct_torsion_relation(self, index_group_atoms, mask_heavy_chains, mask_bridges, seq_len):
        ncacn_dihed = torch.cat(
            [index_group_atoms[:-1, [self.atom_idx['N'],self.atom_idx['CA'],self.atom_idx['C']]], index_group_atoms[1:, [self.atom_idx['N']]]], 
            dim=-1)[mask_bridges]
        cacnca_dihed = torch.cat(
            [index_group_atoms[:-1, [self.atom_idx['CA'],self.atom_idx['C']]], index_group_atoms[1:, [self.atom_idx['N'],self.atom_idx['CA']]]], 
            dim=-1)[mask_bridges]
        cncac_dihed = torch.cat(
            [index_group_atoms[:-1, [self.atom_idx['C']]], index_group_atoms[1:, [self.atom_idx['N'],self.atom_idx['CA'],self.atom_idx['C']]]], 
            dim=-1)[mask_bridges]
        ocnca_dihed = torch.cat(
            [index_group_atoms[:-1, [self.atom_idx['O'],self.atom_idx['C']]], index_group_atoms[1:, [self.atom_idx['N'],self.atom_idx['CA']]]], 
            dim=-1)[mask_bridges]
        
        ncacn_dihed_res_idx = torch.arange(seq_len).to(index_group_atoms)[1:][mask_bridges]
        cacnca_dihed_res_idx = torch.arange(seq_len).to(index_group_atoms)[1:][mask_bridges]
        cncac_dihed_res_idx = torch.arange(seq_len).to(index_group_atoms)[1:][mask_bridges]
        ocnca_dihed_res_idx = torch.arange(seq_len).to(index_group_atoms)[:-1][mask_bridges]
        
        diheds = torch.cat([ncacn_dihed, cacnca_dihed, cncac_dihed, ocnca_dihed], dim=0)
        diheds_res_idx = torch.cat([ncacn_dihed_res_idx, 
                                   cacnca_dihed_res_idx, 
                                   cncac_dihed_res_idx, 
                                   ocnca_dihed_res_idx], dim=0)
        diheds_type = torch.cat([torch.ones(ncacn_dihed.shape[0]) * self.dihed_idx['NCACN'],
                                torch.ones(cacnca_dihed.shape[0]) * self.dihed_idx['CACNCA'],
                                torch.ones(cncac_dihed.shape[0]) * self.dihed_idx['CNCAC'],
                                torch.ones(ocnca_dihed.shape[0]) * self.dihed_idx['OCNCA']]).to(index_group_atoms)
        return diheds, diheds_type, diheds_res_idx
    
    @torch.no_grad()
    def construct_geom_relation(self, segment_ids, segment_idx, S, cdr_range):
        index_atoms = torch.arange(S.shape[0] * 4).to(S)
        index_group_atoms = index_atoms.reshape(-1, 4)
        mask_heavy_chains = (segment_ids == self.BOH_idx)
        mask_heavy_chains[segment_idx] = False
        mask_bridges = torch.logical_and(mask_heavy_chains[:-1], mask_heavy_chains[1:])

        bonds, bonds_type, bonds_res_idx = self.construct_bond_relation(index_group_atoms, mask_heavy_chains, mask_bridges, S.shape[0])
        
        angles, angles_type, angles_res_idx = self.construct_angle_relation(index_group_atoms, mask_heavy_chains, mask_bridges, S.shape[0])
        
        diheds, diheds_type, diheds_res_idx = self.construct_torsion_relation(index_group_atoms, mask_heavy_chains, mask_bridges, S.shape[0])
        
        ncacn_dihed = diheds[torch.where(diheds_type==self.dihed_idx['NCACN'])[0]]
        ncacn_dihed_res_idx = diheds_res_idx[torch.where(diheds_type==self.dihed_idx['NCACN'])[0]]
        cacnca_dihed = diheds[torch.where(diheds_type==self.dihed_idx['CACNCA'])[0]]
        cacnca_dihed_res_idx = diheds_res_idx[torch.where(diheds_type==self.dihed_idx['CACNCA'])[0]]
        cncac_dihed = diheds[torch.where(diheds_type==self.dihed_idx['CNCAC'])[0]]
        cncac_dihed_res_idx = diheds_res_idx[torch.where(diheds_type==self.dihed_idx['CNCAC'])[0]]
        ocnca_dihed = diheds[torch.where(diheds_type==self.dihed_idx['OCNCA'])[0]]
        ocnca_dihed_res_idx = diheds_res_idx[torch.where(diheds_type==self.dihed_idx['OCNCA'])[0]]

        ncac_angle = angles[torch.where(angles_type==self.angle_idx['NCAC'])[0]]
        ncac_angle_res_idx = angles_res_idx[torch.where(angles_type==self.angle_idx['NCAC'])[0]]
        cacn_angle = angles[torch.where(angles_type==self.angle_idx['CACN'])[0]]
        cacn_angle_res_idx = angles_res_idx[torch.where(angles_type==self.angle_idx['CACN'])[0]]
        cnca_angle = angles[torch.where(angles_type==self.angle_idx['CNCA'])[0]]
        cnca_angle_res_idx = angles_res_idx[torch.where(angles_type==self.angle_idx['CNCA'])[0]]
        ocn_angle = angles[torch.where(angles_type==self.angle_idx['OCN'])[0]]
        ocn_angle_res_idx = angles_res_idx[torch.where(angles_type==self.angle_idx['OCN'])[0]]

        nca_bond = bonds[torch.where(bonds_type==self.bond_idx['NCA'])[0]]
        nca_bond_res_idx = bonds_res_idx[torch.where(bonds_type==self.bond_idx['NCA'])[0]]
        cac_bond = bonds[torch.where(bonds_type==self.bond_idx['CAC'])[0]]
        cac_bond_res_idx = bonds_res_idx[torch.where(bonds_type==self.bond_idx['CAC'])[0]]
        co_bond = bonds[torch.where(bonds_type==self.bond_idx['CO'])[0]]
        co_bond_res_idx = bonds_res_idx[torch.where(bonds_type==self.bond_idx['CO'])[0]]
        cn_bond = bonds[torch.where(bonds_type==self.bond_idx['CN'])[0]]
        cn_bond_res_idx = bonds_res_idx[torch.where(bonds_type==self.bond_idx['CN'])[0]]

        ncacn_dihed_cdrs = []
        cacnca_dihed_cdrs = []
        cncac_dihed_cdrs = []
        ocnca_dihed_cdrs = []

        ncac_angle_cdrs = []
        cacn_angle_cdrs = []
        cnca_angle_cdrs = []
        ocn_angle_cdrs = []

        nca_bond_cdrs = []
        cac_bond_cdrs = []
        co_bond_cdrs = []
        cn_bond_cdrs = []

        for start, end in cdr_range:
            ncacn_dihed_cdr = ncacn_dihed[torch.where(ncacn_dihed_res_idx==start)[0]:
                                          torch.where(ncacn_dihed_res_idx==end+1)[0]]
            cacnca_dihed_cdr = cacnca_dihed[torch.where(cacnca_dihed_res_idx==start)[0]:
                                          torch.where(cacnca_dihed_res_idx==end+1)[0]]
            cncac_dihed_cdr = cncac_dihed[torch.where(cncac_dihed_res_idx==start)[0]:
                                          torch.where(cncac_dihed_res_idx==end+1)[0]]
            ocnca_dihed_cdr = ocnca_dihed[torch.where(ocnca_dihed_res_idx==start)[0]:
                                          torch.where(ocnca_dihed_res_idx==end+1)[0]]
            ncacn_dihed_cdrs.append(ncacn_dihed_cdr)
            cacnca_dihed_cdrs.append(cacnca_dihed_cdr)
            cncac_dihed_cdrs.append(cncac_dihed_cdr)
            ocnca_dihed_cdrs.append(ocnca_dihed_cdr)

            
            ncac_angle_cdr = ncac_angle[torch.where(ncac_angle_res_idx==start)[0]:
                                        torch.where(ncac_angle_res_idx==end+1)[0]]
            cacn_angle_cdr = cacn_angle[torch.where(cacn_angle_res_idx==start)[0]:
                                        torch.where(cacn_angle_res_idx==end+1)[0]]
            cnca_angle_cdr = cnca_angle[torch.where(cnca_angle_res_idx==start)[0]:
                                        torch.where(cnca_angle_res_idx==end+1)[0]]
            ocn_angle_cdr = ocn_angle[torch.where(ocn_angle_res_idx==start)[0]:
                                        torch.where(ocn_angle_res_idx==end+1)[0]]
            ncac_angle_cdrs.append(ncac_angle_cdr)
            cacn_angle_cdrs.append(cacn_angle_cdr)
            cnca_angle_cdrs.append(cnca_angle_cdr)
            ocn_angle_cdrs.append(ocn_angle_cdr)

            
            nca_bond_cdr = nca_bond[torch.where(nca_bond_res_idx==start)[0]:
                                    torch.where(nca_bond_res_idx==end+1)[0]]
            cac_bond_cdr = cac_bond[torch.where(cac_bond_res_idx==start)[0]:
                                    torch.where(cac_bond_res_idx==end+1)[0]]
            co_bond_cdr = co_bond[torch.where(co_bond_res_idx==start)[0]:
                                    torch.where(co_bond_res_idx==end+1)[0]]
            cn_bond_cdr = cn_bond[torch.where(cn_bond_res_idx==start)[0]:
                                    torch.where(cn_bond_res_idx==end+1)[0]]
            nca_bond_cdrs.append(nca_bond_cdr)
            cac_bond_cdrs.append(cac_bond_cdr)
            co_bond_cdrs.append(co_bond_cdr)
            cn_bond_cdrs.append(cn_bond_cdr)

        ncacn_dihed_cdrs = torch.cat(ncacn_dihed_cdrs)
        cacnca_dihed_cdrs = torch.cat(cacnca_dihed_cdrs)
        cncac_dihed_cdrs = torch.cat(cncac_dihed_cdrs)
        ocnca_dihed_cdrs = torch.cat(ocnca_dihed_cdrs)

        ncac_angle_cdrs = torch.cat(ncac_angle_cdrs)
        cacn_angle_cdrs = torch.cat(cacn_angle_cdrs)
        cnca_angle_cdrs = torch.cat(cnca_angle_cdrs)
        ocn_angle_cdrs = torch.cat(ocn_angle_cdrs)

        nca_bond_cdrs = torch.cat(nca_bond_cdrs)
        cac_bond_cdrs = torch.cat(cac_bond_cdrs)
        co_bond_cdrs = torch.cat(co_bond_cdrs)
        cn_bond_cdrs = torch.cat(cn_bond_cdrs)


        diheds_cdr = torch.cat([ncacn_dihed_cdrs, cacnca_dihed_cdrs, cncac_dihed_cdrs, ocnca_dihed_cdrs], dim=0)
        angles_cdr = torch.cat([ncac_angle_cdrs, cacn_angle_cdrs, cnca_angle_cdrs, ocn_angle_cdrs], dim=0)
        bonds_cdr = torch.cat([nca_bond_cdrs, cac_bond_cdrs, co_bond_cdrs, cn_bond_cdrs], dim=0)

        diheds_type_cdr = torch.cat([torch.ones(ncacn_dihed_cdrs.shape[0]) * 0,
                                torch.ones(cacnca_dihed_cdrs.shape[0]) * 1,
                                torch.ones(cncac_dihed_cdrs.shape[0]) * 2,
                                torch.ones(ocnca_dihed_cdrs.shape[0]) * 3]).to(S)
        angles_type_cdr = torch.cat([torch.ones(ncac_angle_cdrs.shape[0]) * 0,
                                torch.ones(cacn_angle_cdrs.shape[0]) * 1,
                                torch.ones(cnca_angle_cdrs.shape[0]) * 2,
                                torch.ones(ocn_angle_cdrs.shape[0]) * 3]).to(S)
        bonds_type_cdr = torch.cat([torch.ones(nca_bond_cdrs.shape[0]) * 0,
                                torch.ones(cac_bond_cdrs.shape[0]) * 1,
                                torch.ones(co_bond_cdrs.shape[0]) * 2,
                                torch.ones(cn_bond_cdrs.shape[0]) * 3]).to(S)

        return [bonds, angles, diheds], \
            [bonds_type, angles_type, diheds_type], \
            [bonds_res_idx, angles_res_idx, diheds_res_idx],\
            [bonds_cdr, angles_cdr, diheds_cdr],\
            [bonds_type_cdr, angles_type_cdr, diheds_type_cdr]
    
    def cal_geom_val(self, X, geom_idx_list, geom_type_list):
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
                if self.stable_angle:
                    angles = (r1 * r2).sum(dim=-1)
                else:
                    angles = torch.acos((r1 * r2).sum(dim=-1))
                geom_targets.append(angles)
            elif geom_idx.shape[1] == 2:
                distance = (X[geom_idx[:,0]] - X[geom_idx[:,1]]).norm(dim=-1)
                geom_targets.append(distance)
        return geom_targets
    
    @torch.no_grad()
    def construct_atom_graph(self, aa, atom_edge_index_list, atom_edge_type_list):
        atoms = torch.arange(4).unsqueeze(0).repeat(aa.shape[0], 1).to(aa)
        atom_edge_index = torch.cat([atom_edge_index_list[0],      # relation of bond
                                     atom_edge_index_list[1][:,[0,2]], # relation of angle
                                     atom_edge_index_list[2][:,[0,3]]], # relation of dihedral
                                     dim=0)
        atom_edge_type = torch.cat([atom_edge_type_list[0], 
                                    atom_edge_type_list[1] + len(self.bond_idx), 
                                    atom_edge_type_list[2] + len(self.bond_idx) + len(self.angle_idx)], dim=0)

        return atoms, atom_edge_index, atom_edge_type


    def forward(self, segment_ids, segment_idx, S, cdr_range):
        (intra_geom_idx_list, 
        intra_geom_type_list, 
        intra_geom_res_idx_list, 
        intra_geom_cdr_idx_list, 
        intra_geom_cdr_type_list) = self.construct_geom_relation(segment_ids, segment_idx, S, cdr_range)

        (atoms, 
         atom_edge_index, 
         atom_edge_type) = self.construct_atom_graph(S, intra_geom_idx_list, intra_geom_type_list)
        
        return atoms, atom_edge_index, atom_edge_type,\
              (intra_geom_idx_list, intra_geom_type_list), (intra_geom_cdr_idx_list, intra_geom_cdr_type_list)

    
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

class ProteinFeaturizer(nn.Module):

    def __init__(self, interface_only, use_esm=False):
        super().__init__()
        # global nodes and mask nodes
        self.boa_idx = VOCAB.symbol_to_idx(VOCAB.BOA)
        self.boh_idx = VOCAB.symbol_to_idx(VOCAB.BOH)
        self.bol_idx = VOCAB.symbol_to_idx(VOCAB.BOL)

        # segment ids
        self.ag_seg_id, self.hc_seg_id, self.lc_seg_id = 1, 2, 3

        # positional embedding
        self.node_pos_embedding = PosEmbedding(16)
        self.edge_pos_embedding = PosEmbedding(16)
        self.interface_only = interface_only
        self.use_esm = use_esm

    def _construct_segment_ids(self, S):
        # construct segment ids. 1/2/3 for antigen/heavy chain/light chain
        glbl_node_mask = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx)
        glbl_nodes = S[glbl_node_mask]
        boa_mask, boh_mask, bol_mask = (glbl_nodes == self.boa_idx), (glbl_nodes == self.boh_idx), (glbl_nodes == self.bol_idx)
        glbl_nodes[boa_mask], glbl_nodes[boh_mask], glbl_nodes[bol_mask] = self.ag_seg_id, self.hc_seg_id, self.lc_seg_id
        segment_ids = torch.zeros_like(S)
        segment_ids[glbl_node_mask] = glbl_nodes - F.pad(glbl_nodes[:-1], (1, 0), value=0)
        segment_ids = torch.cumsum(segment_ids, dim=0)

        segment_idx = torch.zeros_like(S)
        segment_idx[glbl_node_mask] = 1.0
        segment_mask = torch.cumsum(segment_idx, dim=0)

        return segment_ids, segment_mask, torch.nonzero(segment_idx)[:, 0]

    def _radial_edges(self, X, src_dst, cutoff):
        dist = X[:, 1][src_dst]  # [Ef, 2, 3], CA position
        dist = torch.norm(dist[:, 0] - dist[:, 1], dim=-1) # [Ef]
        src_dst = src_dst[dist <= cutoff]
        src_dst = src_dst.transpose(0, 1)  # [2, Ef]
        return src_dst

    def _knn_edges(self, X, offsets, segment_ids, is_global, top_k=5, eps=1e-6):

        for batch in range(len(offsets)):
            if batch != len(offsets) - 1:
                X_batch = X[offsets[batch]:offsets[batch+1], 1, :]
            else:
                X_batch = X[offsets[batch]:, 1, :]

            dX = torch.unsqueeze(X_batch, 0) - torch.unsqueeze(X_batch, 1)
            D = torch.sqrt(torch.sum(dX**2, 2) + eps)
            _, E_idx = torch.topk(D, top_k, dim=-1, largest=False)

            if batch == 0:
                row = torch.arange(E_idx.shape[0], device=X.device).view(-1, 1).repeat(1, top_k).view(-1)
                col = E_idx.view(-1)
            else:
                row = torch.cat([row, torch.arange(E_idx.shape[0], device=X.device).view(-1, 1).repeat(1, top_k).view(-1) + offsets[batch]], dim=0)
                col = torch.cat([col, E_idx.view(-1) + offsets[batch]], dim=0)
         
        row_seg, col_seg = segment_ids[row], segment_ids[col]
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))

        select_edges = torch.logical_and(row_seg == col_seg, not_global_edges)
        ctx_edges_knn = torch.stack([row[select_edges], col[select_edges]])

        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_edges_knn = torch.stack([row[select_edges], col[select_edges]])

        return ctx_edges_knn, inter_edges_knn
    
    def edge_masking(self, pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats, edge_type):
        if edge_type == 6 or edge_type == 7:
            pos_edge_feats *= 0

        if edge_type == 1 or edge_type == 2:
            pos_edge_feats *= 0
            # dis_edge_feats *= 0
            angle_edge_feats *= 0
            direct_edge_feats *= 0

        return pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats

    def get_node_pos(self, X, segment_mask, segment_idx):
        pos = torch.arange(X.shape[0], device=X.device) - segment_idx[segment_mask-1]
        pos_node_feats = self.node_pos_embedding(pos.view(1, X.shape[0], 1))[0, :, 0, :]  # [1, N, 1, 16] -> [N, 16]

        return pos_node_feats
    
    def _rbf(self, D):
        D_min, D_max, D_count = 0., 20., 16
        D_mu = torch.linspace(D_min, D_max, D_count).cuda()
        D_mu = D_mu.view([1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        return RBF
    
    def get_node_dist(self, X, eps=1e-6):
        d_NC = torch.sqrt(torch.sum((X[:, 0, :] - X[:, 1, :])**2, dim=1) + eps)
        d_CC = torch.sqrt(torch.sum((X[:, 2, :] - X[:, 1, :])**2, dim=1) + eps)
        d_OC = torch.sqrt(torch.sum((X[:, 3, :] - X[:, 1, :])**2, dim=1) + eps)

        d_NC_RBF = self._rbf(d_NC)
        d_CC_RBF = self._rbf(d_CC)
        d_OC_RBF = self._rbf(d_OC)

        dis_node_feats = torch.cat((d_NC_RBF, d_CC_RBF, d_OC_RBF), 1)

        return dis_node_feats

    def get_node_angle(self, X, segment_idx, segment_ids, eps=1e-6):

        # First 3 coordinates are N, CA, C
        X = X[:, :3,:].reshape(1, 3*X.shape[0], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]

        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)

        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        D = F.pad(D, (1,2), 'constant', 0)

        # psi, omega, phi
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        Dihedral_Angle_features  = torch.cat((torch.cos(D), torch.sin(D)), 2)


        cosD = (u_2*u_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.acos(cosD)
        D = F.pad(D, (1,2), 'constant', 0)

        # beta, alpha, gamma
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

        angle_node_feats = torch.cat((Dihedral_Angle_features, Angle_features), 2)[0]

        for i in segment_idx:
            if i == 0:
                angle_node_feats[:, i:i+2] = 0
            else:
                angle_node_feats[:, i-1:i+2] = 0

        if self.interface_only == 0:
            angle_node_feats[segment_ids == self.ag_seg_id] = 0

        return angle_node_feats

    def get_node_direct(self, Xs, segment_idx, segment_ids):
        X = Xs[:, 1,:].reshape(1, Xs.shape[0], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:] # # CA-N, C-CA, N-C, CA-N ...
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]

        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        o_1 = F.normalize(u_2 - u_1, dim=-1)

        # Build relative orientations
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0,0,1,2), 'constant', 0)
        O = O.view(list(O.shape[:2]) + [3,3])

        for i in segment_idx:
            if i == 0:
                O[:, i:i+2, :, :] = 0
            else:
                O[:, i-2:i+2, :, :] = 0

        # Rotate into local reference frames
        d_NC = (Xs[:, 0, :] - Xs[:, 1, :]).reshape(1, Xs.shape[0], 3, 1)
        d_NC = F.normalize(torch.matmul(O, d_NC).squeeze(-1), dim=-1)
        
        d_CC = (Xs[:, 2, :] - Xs[:, 1, :]).reshape(1, Xs.shape[0], 3, 1)
        d_CC = F.normalize(torch.matmul(O, d_CC).squeeze(-1), dim=-1)

        d_OC = (Xs[:, 3, :] - Xs[:, 1, :]).reshape(1, Xs.shape[0], 3, 1)
        d_OC = F.normalize(torch.matmul(O, d_OC).squeeze(-1), dim=-1)

        direct_node_feats = torch.cat((d_NC, d_CC, d_OC), 2)[0]

        if self.interface_only == 0:
            direct_node_feats[segment_ids == self.ag_seg_id] = 0

        return direct_node_feats, O

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
        """

        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes

        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        return Q

    def get_edge_pos(self, edge_index):
        pos = (edge_index[0:1, :] - edge_index[1:2, :]).float().unsqueeze(-1)
        pos_edge_feats = self.edge_pos_embedding(pos)[0, :, 0, :]  # [1, E, 1, 16] -> [E, 16]

        return pos_edge_feats

    def get_edge_dist(self, X, edge_index, eps=1e-6):
        X_row, X_col = X[edge_index[0, :]], X[edge_index[1, :]]

        d_NC = torch.sqrt(torch.sum((X_row[:, 0, :] - X_col[:, 1, :])**2, dim=1) + eps)
        d_CAC = torch.sqrt(torch.sum((X_row[:, 1, :] - X_col[:, 1, :])**2, dim=1) + eps)
        d_CC = torch.sqrt(torch.sum((X_row[:, 2, :] - X_col[:, 1, :])**2, dim=1) + eps)
        d_OC = torch.sqrt(torch.sum((X_row[:, 3, :] - X_col[:, 1, :])**2, dim=1) + eps)
        
        d_NC_RBF = self._rbf(d_NC)
        d_CAC_RBF = self._rbf(d_CAC)
        d_CC_RBF = self._rbf(d_CC)
        d_OC_RBF = self._rbf(d_OC)

        dis_edge_feats = torch.cat((d_NC_RBF, d_CAC_RBF, d_CC_RBF, d_OC_RBF), 1)

        return dis_edge_feats

    def get_edge_angle(self, O, edge_index):

        O_row, O_col = O[:, edge_index[0, :], :, :].unsqueeze(2), O[:, edge_index[1, :], :, :].unsqueeze(2)
        R = torch.matmul(O_row.transpose(-1,-2), O_col)
        angle_edge_feats = self._quaternions(R)[0, :, 0, :]
        
        return angle_edge_feats

    def get_edge_direct(self, X, O, edge_index):
        X_row, X_col = X[edge_index[0, :]], X[edge_index[1, :]]
        _, O_col = O[:, edge_index[0, :], :, :], O[:, edge_index[1, :], :, :]

        # Rotate into local reference frames
        d_NC = (X_row[:, 0, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_NC = F.normalize(torch.matmul(O_col, d_NC).squeeze(-1), dim=-1)
        
        d_CAC = (X_row[:, 1, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_CAC = F.normalize(torch.matmul(O_col, d_CAC).squeeze(-1), dim=-1)

        d_CC = (X_row[:, 2, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_CC = F.normalize(torch.matmul(O_col, d_CC).squeeze(-1), dim=-1)

        d_OC = (X_row[:, 3, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_OC = F.normalize(torch.matmul(O_col, d_OC).squeeze(-1), dim=-1)

        direct_edge_feats = torch.cat((d_NC, d_CAC, d_CC, d_OC), 2)[0]

        return direct_edge_feats

    @torch.no_grad()
    def construct_edges(self, X, S, batch_id, segment_ids=None, ESMs=None):
        '''
        Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch
        '''
        # construct tensors to map between global / local node index
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs]
        
        # global node index to local index. lni2gni can be implemented as lni + offsets[batch_id]
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]  # [N]

        # prepare inputs
        if segment_ids is None:
            segment_ids, segment_mask, segment_idx = self._construct_segment_ids(S)

        # all possible edges (within the same graph)
        # same bid (get rid of self-loop and none edges)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)

        # shift right and pad 1 to the left
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0  # delete self loop
        row, col = torch.nonzero(same_bid).T  # [2, n_edge_all]
        col = col + offsets[batch_id[row]]  # mapping from local to global node index
        
        # not global edges
        is_global = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx) # [N]
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))

        # all possible ctx edges: same seg, not global
        row_seg, col_seg = segment_ids[row], segment_ids[col]
        select_edges = torch.logical_and(row_seg == col_seg, not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]

        ctx_edges_rball = self._radial_edges(X, torch.stack([ctx_all_row, ctx_all_col]).T, cutoff=8.0)
        ctx_edges_knn, inter_edges_knn = self._knn_edges(X, offsets, segment_ids, is_global, top_k=8)

        if self.interface_only == 0:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 1, (row - col) == -1), select_edges, row_seg != self.ag_seg_id)
        else:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 1, (row - col) == -1), select_edges)
        ctx_edges_seq_d1 = torch.stack([row[select_edges_seq], col[select_edges_seq]])

        if self.interface_only == 0:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 2, (row - col) == -2), select_edges, row_seg != self.ag_seg_id)
        else:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 2, (row - col) == -2), select_edges)
        ctx_edges_seq_d2 = torch.stack([row[select_edges_seq], col[select_edges_seq]])

        # all possible inter edges: not same seg, not global
        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        inter_edges_rball = self._radial_edges(X, torch.stack([inter_all_row, inter_all_col]).T, cutoff=12.0)

        # edges between global and normal nodes
        select_edges = torch.logical_and(row_seg == col_seg, torch.logical_not(not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]

        # edges between global and global nodes
        select_edges = torch.logical_and(row_global, col_global) # self-loop has been deleted
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]

        # construct node features
        pos_node_feats = self.get_node_pos(X, segment_mask, segment_idx)
        dis_node_feats = self.get_node_dist(X)
        angle_node_feats = self.get_node_angle(X, segment_idx.tolist(), segment_ids)
        direct_node_feats, O = self.get_node_direct(X, segment_idx.tolist(), segment_ids)
        node_feats = torch.cat((pos_node_feats, dis_node_feats, angle_node_feats, direct_node_feats), 1)
        if hasattr(self, 'use_esm'):
            node_feats = torch.cat((node_feats, ESMs), dim=1) if self.use_esm else node_feats
        # construct edge features
        edges_list = [ctx_edges_rball, global_normal, global_global, ctx_edges_seq_d1, ctx_edges_knn, ctx_edges_seq_d2, inter_edges_rball, inter_edges_knn]
        edge_class_type = torch.eye(len(edges_list), dtype=torch.float, device=X.device)
        edge_feats_list = []

        for i in range(len(edges_list)):
            type_edge_feats = edge_class_type[torch.ones(edges_list[i].shape[1]).long() * i]
            pos_edge_feats = self.get_edge_pos(edges_list[i])
            dis_edge_feats = self.get_edge_dist(X, edges_list[i])
            angle_edge_feats = self.get_edge_angle(O, edges_list[i])
            direct_edge_feats = self.get_edge_direct(X, O, edges_list[i])
            pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats = self.edge_masking(pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats, i)
            edge_feats = torch.cat((type_edge_feats, pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats), 1)
            edge_feats_list.append(edge_feats)       

        return edges_list, edge_feats_list, node_feats, segment_ids, segment_idx

    def forward(self, X, S, offsets, ESMs=None):
        batch_id = torch.zeros_like(S)
        batch_id[offsets[1:-1]] = 1
        batch_id = torch.cumsum(batch_id, dim=0)

        return self.construct_edges(X, S, batch_id, ESMs=ESMs)
