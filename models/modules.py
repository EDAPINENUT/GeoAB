import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .geometry import BondLength, BondAngle, TorsionAngle
import math
from torch_geometric.utils import softmax, scatter

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
    

class BondLengthMechanics(nn.Module):

    def __init__(self, dim_node_embedding: int, num_layer: int=2, num_bond_token:int=4, dim_bond_embedding:int=64):
        super().__init__()
        self.bond_encoder = nn.Embedding(num_bond_token, dim_bond_embedding)
        self.length_encoder = Smearing(dim_embedding=dim_bond_embedding)
        self.edge_model = LeakyMLP(dim_node_embedding * 2 + dim_bond_embedding * 2, dim_node_embedding, num_layer=num_layer, dim_end=2)

    def force(self, x: Tensor, bond_index: Tensor, node_embedding: Tensor, bond_types: Tensor) -> Tensor:
        geometry = BondLength(bond_index=bond_index, coords=x)
        force_direction = geometry.force_direction()
        distance = geometry.geom()
        force_norm = self.edge_model(torch.cat([
            node_embedding[bond_index[:,0]],
            node_embedding[bond_index[:,1]],
            self.bond_encoder(bond_types.long()),
            self.length_encoder(distance)
            ], dim=-1))

        return force_norm, force_direction

    def forward(self, x: Tensor, bond_index: Tensor, node_embedding: Tensor, pos: Tensor, bond_types: Tensor):
        force, direction = self.force(x, bond_index, node_embedding, bond_types)
        pos = self.update_pos(pos, bond_index, force, direction) #, step_size=0.5 * sigma
        return pos
    
    def update_pos(self, pos: Tensor, bond_index: Tensor, force: Tensor, direction: Tensor, step_size=0.5):
        update = force.unsqueeze(dim=-2) * direction
        if torch.is_tensor(step_size):
            step_size0 = step_size[None, :, None].expand_as(pos)[bond_index[:,0]]
            step_size1 = step_size[None, :, None].expand_as(pos)[bond_index[:,1]]
        else:
            step_size0, step_size1 = step_size, step_size
        pos[bond_index[:,0]] += step_size0 * update[...,0]
        pos[bond_index[:,1]] += step_size1 * update[...,1]
        return pos


def compute_rotation_matrix(theta, d):
    x, y, z = torch.unbind(d, dim=-1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    ret = torch.stack((
        cos + (1 - cos) * x * x,
        (1 - cos) * x * y - sin * z,
        (1 - cos) * x * z + sin * y,
        (1 - cos) * x * y + sin * z,
        cos + (1 - cos) * y * y,
        (1 - cos) * y * z - sin * x,
        (1 - cos) * x * z - sin * y,
        (1 - cos) * y * z + sin * x,
        cos + (1 - cos) * z * z,
    ), dim=-1)
    size = ret.size()[:-1]
    return ret.reshape(*size, 3, 3)  


class BondAngleMechanics(nn.Module):

    def __init__(self, dim_node_embedding: int, num_layer: int=2, dim_angle_embedding:int=64, num_angle_token: int=4):
        super().__init__()
        # self.angle_encoder = TrigoEmbedding(dim_embedding=2, half_encode=True) 
        # TODO # add angle as positional encoding will cause gradient exploding because arcos function. -> use cos(\theta) to replace
        self.angle_model = LeakyMLP(dim_node_embedding*3 + dim_angle_embedding, dim_node_embedding, num_layer=num_layer, dim_end=2)
        self.angle_encoder = nn.Embedding(num_angle_token, dim_angle_embedding)


    def force(self, x: Tensor, angle_index: Tensor, node_embedding: Tensor, angle_types: Tensor) -> Tensor:
        geometry = BondAngle(angle_index=angle_index, coords=x)
        force_direction = geometry.force_direction()
        d1, d2 = geometry.d1, geometry.d2
        # cos_angle = geometry.cos_angle
        force_norm = self.angle_model(torch.cat([
            node_embedding[angle_index[:,0]],
            node_embedding[angle_index[:,1]],
            node_embedding[angle_index[:,2]],
            self.angle_encoder(angle_types.long())
            # cos_angle
            ], dim=-1)) # / torch.cat([d1, d2], dim=-1)

        return force_norm, force_direction

    def forward(self, x: Tensor, angle_index: Tensor, node_embedding: Tensor, pos: Tensor, angle_types: Tensor):
        force, direction = self.force(x, angle_index, node_embedding, angle_types)
        # update = force.unsqueeze(dim=-2) * direction
        pos = self.update_pos(pos, angle_index, force, direction)# step_size=0.5 * sigma)
        return pos

    def update_pos(self, pos: Tensor, angle_index: Tensor, force: Tensor, direction: Tensor, step_size=0.5):
        theta_0 = force[..., 0]
        theta_2 = force[..., 1]
        if torch.is_tensor(step_size):
            step_size0 = step_size[None, :].expand_as(theta_0)[angle_index[:,0]]
            step_size2 = step_size[None, :].expand_as(theta_2)[angle_index[:,2]]
        else:
            step_size0, step_size2 = step_size, step_size

        rot_0 = compute_rotation_matrix(theta_0 * step_size0, d=direction[..., 0])
        rot_2 = compute_rotation_matrix(theta_2 * step_size2, d=direction[..., 1])
        x_1 = pos[angle_index[:, 1]]

        r_0 = pos[angle_index[:, 0]] - x_1
        r_0_rot = torch.einsum('bij,bjk->bik', rot_0, r_0.unsqueeze(-1)).squeeze(-1)

        r_2 = pos[angle_index[:, 2]] - x_1
        r_2_rot = torch.einsum('bij,bjk->bik', rot_2, r_2.unsqueeze(-1)).squeeze(-1)
        x_0, x_2 = r_0_rot + x_1, r_2_rot + x_1 

        x = torch.cat([x_0, x_1, x_2], dim=0)
        index = torch.cat([angle_index[:,0], angle_index[:,1], angle_index[:,2]], dim=0)
        update_mask = torch.isin(torch.arange(0, pos.shape[0], device=pos.device), index.unique()).unsqueeze(1)
        pos = torch.where(update_mask,
                          unsorted_segment_mean(x, index, pos.shape[0]),
                          pos)

        return pos


class TorsionAngleMechanics(nn.Module):

    def __init__(self, dim_node_embedding: int, num_layer: int=2, num_torsion_token:int=4, dim_torsion_embedding:int=64):
        super().__init__()
        self.torsion_encoder = nn.Embedding(num_torsion_token, dim_torsion_embedding)
        self.angle_encoder = TrigoEmbedding(dim_embedding=2)
        self.torsion_model = LeakyMLP(dim_node_embedding*4 + 2 + dim_torsion_embedding, dim_node_embedding, num_layer=num_layer, dim_end=2)

    def force(self, x: Tensor, torsion_index: Tensor, node_embedding: Tensor, dihed_types: Tensor) -> Tensor:
        geometry = TorsionAngle(torsion_index=torsion_index, coords=x)
        force_direction = geometry.force_direction()
        phi = geometry.geom()
        # d1, d3 = geometry.d1, geometry.d3
        force_norm = self.torsion_model(torch.cat([
            node_embedding[torsion_index[:,0]],
            node_embedding[torsion_index[:,1]],
            node_embedding[torsion_index[:,2]],
            node_embedding[torsion_index[:,3]],
            self.angle_encoder(phi),
            self.torsion_encoder(dihed_types.long())
            ], dim=-1)) # /torch.cat([d1, d3], dim=-1)

        return force_norm, force_direction

    def forward(self, x: Tensor, torsion_index: Tensor, node_embedding: Tensor, pos: Tensor, dihed_types: Tensor):
        force, direction = self.force(x, torsion_index, node_embedding, dihed_types)
        pos = self.update_pos(pos, torsion_index, force, direction) 
        return pos
    
    def update_pos(self, pos: Tensor, torsion_index: Tensor, force: Tensor, direction: Tensor, step_size=0.5):
        theta_0 = force[..., 0]
        theta_3 = force[..., 1]
        if torch.is_tensor(step_size):
            step_size0 = step_size[None, :].expand_as(theta_0)[torsion_index[:,0]]
            step_size3 = step_size[None, :].expand_as(theta_3)[torsion_index[:,3]]
        else:
            step_size0, step_size3 = step_size, step_size

        rot_0 = compute_rotation_matrix(theta_0 * step_size0, d=direction[..., 0])
        rot_3 = compute_rotation_matrix(theta_3 * step_size3, d=direction[..., 1])
        x_0 = pos[torsion_index[:, 0]]
        x_1 = pos[torsion_index[:, 1]]
        x_2 = pos[torsion_index[:, 2]]
        x_3 = pos[torsion_index[:, 3]]

        r_0 = x_0 - x_1
        r_0_rot = torch.einsum('bij,bjk->bik', rot_0, r_0.unsqueeze(-1)).squeeze(-1)

        r_3 = x_3 - x_2
        r_3_rot = torch.einsum('bij,bjk->bik', rot_3, r_3.unsqueeze(-1)).squeeze(-1)
        x_0_update, x_3_update = r_0_rot + x_1, r_3_rot + x_2 

        x = torch.cat([x_0_update, x_1, x_2, x_3_update], dim=0)
        index = torch.cat([torsion_index[:,0], torsion_index[:,1], torsion_index[:,2], torsion_index[:,3]], dim=0)
        update_mask = torch.isin(torch.arange(0, pos.shape[0], device=pos.device), index.unique()).unsqueeze(1)
        pos = torch.where(update_mask,
                          unsorted_segment_mean(x, index, pos.shape[0]),
                          pos)

        return pos


def coord2radial(edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]  # [n_edge, n_channel, d]
    radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))  # [n_edge, n_channel, n_channel]
    # normalize radial
    radial = F.normalize(radial, dim=0)  # [n_edge, n_channel, n_channel]
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class RelationMPNN(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, dropout=0.1, edges_in_d=1, edge_type=8):
        super(RelationMPNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.coord_mlp = nn.ModuleList()
        self.relation_mlp = nn.ModuleList()

        self.message_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + n_channel**2 + edges_in_d, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU())

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_nf + edges_in_d + hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, edges_in_d))
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, output_nf))
         
        for _ in range(edge_type):
            self.relation_mlp.append(nn.Linear(input_nf, input_nf, bias=False))

            layer = nn.Linear(hidden_nf, n_channel, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

            self.coord_mlp.append(nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                nn.SiLU(),
                layer
            ))

    def message_model(self, source, target, radial, edge_attr):
        radial = radial.reshape(radial.shape[0], -1)
        out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.message_mlp(out)
        out = self.dropout(out)

        return out

    def node_model(self, x, edge_list, edge_feat_list):
        agg = self.relation_mlp[0](unsorted_segment_sum(edge_feat_list[0], edge_list[0][0], num_segments=x.size(0)))
        for i in range(1, len(edge_list)):
            agg += self.relation_mlp[i](unsorted_segment_sum(edge_feat_list[i], edge_list[i][0], num_segments=x.size(0)))
            
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        out = self.dropout(out)
        out = x + out

        return out
    
    def coord_model(self, coord, edge_list, edge_feat_list, coord_diff_list):
        tran_list = []
        row_list = []
        for i in range(len(edge_list)):
            trans = coord_diff_list[i] * self.coord_mlp[i](edge_feat_list[i]).unsqueeze(-1)  # [n_edge, n_channel, d]
            tran_list.append(trans)
            row_list.append(edge_list[i][0])
        agg = unsorted_segment_mean(torch.cat(tran_list, dim=0), torch.cat(row_list, dim=0), num_segments=coord.size(0))  # [bs * n_node, n_channel, d]
        coord = coord + agg

        return coord

    def edge_model(self, h, edge_list, edge_feat_list):
        m = []

        for i in range(len(edge_list)):
            row, col = edge_list[i]
            out = torch.cat([h[row], edge_feat_list[i], h[col]], dim=1)
            out = self.edge_mlp(out)
            m.append(out)

        return m
    
    def forward(self, h, coord, edge_attr, edge_list):

        edge_feat_list = []
        coord_diff_list = []

        for i in range(len(edge_list)):
            radial, coord_diff = coord2radial(edge_list[i], coord)
            coord_diff_list.append(coord_diff)

            row, col = edge_list[i]
            edge_feat = self.message_model(h[row], h[col], radial, edge_attr[i])
            edge_feat_list.append(edge_feat)

        h = self.node_model(h, edge_list, edge_feat_list)
        x = self.coord_model(coord, edge_list, edge_feat_list, coord_diff_list)
        m = self.edge_model(h, edge_list, edge_attr)

        return h, x, m


class GraphTransformer(nn.Module):
    def __init__(
        self, 
        dim_input: int,
        dim_embed: int,
        num_head: int, 
        dim_ffn: int = 0,
        dropout: float = 0.5,
        add_self_loops: bool = False,
        leaky_negative_slope: float = 0.2,
        bi: bool = True
    ):
        super().__init__()

        self.add_self_loops = add_self_loops

        self.attention = GraphAttention(
            in_channels=dim_input, 
            mid_channels=dim_embed, 
            out_channels=dim_embed,
            num_head=num_head, 
            dropout=dropout,
            leaky_negative_slope=leaky_negative_slope,
            add_self_loops=add_self_loops,
            bi=bi
        )

        dim_ffn = dim_input if dim_ffn == 0 else dim_ffn

        self.feed_forward = nn.Sequential(
            nn.Linear(dim_embed if add_self_loops else dim_embed+dim_input, dim_ffn),
            nn.GELU(),
            nn.Linear(dim_ffn, dim_embed)
        )
    
    def forward(self, node_attr: Tensor, edge_index: Tensor , edge_attr: Tensor=None) -> Tensor:

        node_embeding = self.attention(node_attr, edge_index, edge_attr)
        if self.add_self_loops:
            return self.feed_forward(node_embeding)
        else:
            return self.feed_forward(torch.cat([node_embeding,node_attr],dim=1))

        

class GraphAttention(nn.Module):
    """Graph self-attention layer (V2)
    Brody, Shaked, Uri Alon, and Eran Yahav. “How Attentive Are Graph Attention Networks?” 
    ArXiv:2105.14491 [Cs], January 31, 2022. http://arxiv.org/abs/2105.14491.
    """
    def __init__(
        self, 
        in_channels: int,
        mid_channels: int,
        out_channels: int = None,
        num_head: int = 1, 
        dropout: float = 0.5,
        leaky_negative_slope: float = 0.2,
        add_self_loops : bool = False,
        concat: bool = True,
        share_weights: bool = False,
        bi: bool = True
    ):
        super().__init__()

        self.num_head = num_head
        self.dim_head, remainder = divmod(mid_channels, num_head)

        self.mid_channels = mid_channels
        self.in_channels = in_channels
        
        out_channels = out_channels if out_channels is not None else in_channels
        self.out_channels = out_channels

        if remainder != 0:
            raise Exception("Multihead attention requires a dimension divisible by the number of heads")

        self.lin_l = nn.Linear(in_channels, mid_channels, bias=False)

        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = nn.Linear(in_channels, mid_channels, bias=False)

        self.lin_s = nn.Linear(in_channels, num_head) if add_self_loops else None
        self.att = nn.Parameter(torch.Tensor(1, num_head, self.dim_head))
        torch.nn.init.xavier_normal_(self.att)
        self.activation = nn.LeakyReLU(negative_slope=leaky_negative_slope)
        self.dropout = nn.Dropout(dropout)

        self.concat = concat
        self.bi = bi

    def forward(self, node_attr: Tensor, index: Tensor, edge_attr: Tensor = None) -> Tensor:
    
        N = node_attr.shape[0]

        ngl  = self.lin_l(node_attr).view(N,self.num_head,self.dim_head)
        ngr  = self.lin_r(node_attr).view(N,self.num_head,self.dim_head)
        if edge_attr is not None:
            edge_attr = edge_attr.view(-1, self.num_head,self.dim_head)

        if self.bi:
            index = torch.cat([index, index[:,[1,0]]], dim=0)
            if edge_attr is not None:
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)


        #
        # Construct att_weight
        #
        gl = ngl[index[:,0]]
        gr = ngr[index[:,1]]
        if edge_attr is not None:
            att_weight = (self.att * self.activation(gl + gr + edge_attr)).sum(-1)
        else:
            att_weight = (self.att * self.activation(gl + gr)).sum(-1)
        
        if not self.lin_s is None:
            index = torch.cat([
                index,
                torch.arange(N,device=index.device).unsqueeze(-1).expand(N,2)
            ],dim=0)

            att_weight = torch.cat([
                att_weight,
                self.lin_s(node_attr)
            ],dim=0)

            gr = torch.cat([
                gr,
                ngr
            ])

        ind = index[:,0]
        
        alpha = self.dropout(softmax(src=att_weight, index=ind, num_nodes=N, dim=0))

        message = gr * alpha.unsqueeze(-1)
        # ind = index[:,0].view(index.shape[0],1,1).expand(-1,message.shape[1],message.shape[2])
        out = scatter(message, index = index[:,0], reduce='sum', dim_size=N)

        if self.concat:
            out = out.view(-1, self.num_head * self.dim_head)
        else:
            out = out.mean(dim=1)
        
        return out



    def __repr__(self):
        return '{}({}, {}, num_head={})'.format(self.__class__.__name__,
                                                self.in_channels,
                                                self.mid_channels, 
                                                self.num_head)

class TransformerEncoder(nn.Module):

    def __init__(self, node_input_dim: int, dim_embed: int, 
                 dim_ffn: int=None, num_head: int=5, 
                 num_layer: int=3, 
                 dropout: float=0.5, edge_input_dim: int=6,
                 bi: bool=True):
        super().__init__()

        dim_ffn = dim_ffn if dim_ffn is not None else dim_embed * 4
        self.node_encoder = nn.Linear(node_input_dim, dim_embed)
        self.edge_encoder = nn.Linear(edge_input_dim, dim_embed)

        self.normalization = 1 / dim_embed**0.5

        self.layers = nn.ModuleList([
            GraphTransformer(dim_embed, 
                             dim_embed, 
                             dim_ffn=dim_ffn, 
                             num_head=num_head, 
                             add_self_loops=False, 
                             dropout=dropout,
                             bi=bi) 
                             for _ in range(num_layer)
            ])

    def forward(self, atom_type, edge_type, edge_index) -> Tensor:

        node_emb = self.node_encoder(atom_type)
        
        edge_emb = self.edge_encoder(edge_type)

        for layer in self.layers:
            node_emb = layer(node_emb, edge_index, edge_emb)
        return node_emb, edge_emb


class AtomEncoder(nn.Module):
    def __init__(self, dim_embed, num_layer, dropout=0.1, dim_ffn=None, num_head=8, num_layer_mlp=2) -> None:
        super().__init__()
        self.dim_embed = dim_embed
        self.atom_embed = nn.Embedding(4, self.dim_embed)
        self.edge_embed = nn.Embedding(20, self.dim_embed)
        dim_ffn = 2*self.dim_embed if dim_ffn is None else dim_ffn

        self.num_layer = num_layer
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

    
    def forward(self, pos, aa, aa_embedding, bonds, bond_types, angles, angle_types, torsions, torsion_types) -> Tensor:
        atoms = torch.arange(4).unsqueeze(0).repeat(aa.shape[0], 1).to(aa)
        pos = pos.view(atoms.shape[0]*atoms.shape[1], -1)
        edge_index = torch.cat([bonds,      # relation of bond
                                angles[:,[0,2]], # relation of angle
                                torsions[:,[0,3]]], # relation of dihedral
                                dim=0)
        edge_type = torch.cat([bond_types, 
                               angle_types + bond_types.max() + 1 , 
                               torsion_types + angle_types.max() + bond_types.max() + 2], dim=0)

        node_embedding = (self.atom_embed(atoms) + aa_embedding.unsqueeze(1)).view(atoms.shape[0]*atoms.shape[1], -1)
        edge_embedding = self.edge_embed(edge_type)
        node_embedding, _ = self.gnn_atom(node_embedding, edge_embedding, edge_index)
        return node_embedding

class RelationEGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, n_layers=4, dropout=0.1, node_feats_dim=0, edge_feats_dim=1):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf + node_feats_dim, hidden_nf)
        self.linear_out = nn.Linear(hidden_nf, out_node_nf)

        for i in range(n_layers):
            self.add_module(f'layer_{i}', RelationMPNN(hidden_nf, hidden_nf, hidden_nf, n_channel, dropout=dropout, edges_in_d=edge_feats_dim))
    
    def forward(self, h, x, edges_list, edge_feats_list, node_feats):
        h = torch.cat((h, node_feats), 1)
        h = self.linear_in(h)
        h = self.dropout(h)

        m = edge_feats_list
        for i in range(self.n_layers):
            h, x, m = self._modules[f'layer_{i}'](h, x, m, edges_list)

        h = self.dropout(h)
        logit = self.linear_out(h)

        return logit, x, h