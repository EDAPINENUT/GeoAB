import torch
import torch.nn.functional as F

class BondLength():
    def __init__(self, bond_index, coords) -> None:
        self.bond_index = bond_index
        self.coords = coords
        self.r = torch.index_select(coords,0,self.bond_index[:,0]) - torch.index_select(coords,0,self.bond_index[:,1])
        d_square = torch.square(self.r).sum(-1, keepdim=True)
        self.d = torch.sqrt(d_square.clamp(min=1e-7))
        self.unit_vec = self.r / self.d
    
    def geom(self):
        return self.d
    
    def force_direction(self):
        return torch.stack([-self.unit_vec, self.unit_vec], dim=-1)

class BondAngle():
    def __init__(self, angle_index, coords) -> None:
        self.angle_index = angle_index
        self.coords = coords
        self.r1 = torch.index_select(coords,0,self.angle_index[:,0]) - torch.index_select(coords,0,self.angle_index[:,1])
        self.r2 = torch.index_select(coords,0,self.angle_index[:,2]) - torch.index_select(coords,0,self.angle_index[:,1])
        self.d1 = torch.norm(self.r1, dim=-1, keepdim=True)
        self.d2 = torch.norm(self.r2, dim=-1, keepdim=True)
        inner_product = (self.r1 * self.r2).sum(dim=-1, keepdim=True)
        self.cos_angle = inner_product / (self.d1 * self.d2)
        self.theta = torch.acos(self.cos_angle)

        self.omega = torch.cross(self.r1, self.r2)
        self.omega_unit = F.normalize(self.omega, p=2, dim=-1)
    
    def geom(self):
        return self.theta
    
    def force_direction(self):
        return torch.stack([-self.omega_unit, self.omega_unit], dim=-1)

        # return torch.stack([-self.unit_vec, self.unit_vec], dim=-1)


class TorsionAngle():
    def __init__(self, torsion_index, coords) -> None:
        r1 = coords[torsion_index[:,1],] - coords[torsion_index[:,0],]
        r2 = coords[torsion_index[:,2],] - coords[torsion_index[:,1],]
        r3 = coords[torsion_index[:,3],] - coords[torsion_index[:,2],]

        self.d1 = torch.norm(r1, dim=-1, keepdim=True)
        self.d3 = torch.norm(r3, dim=-1, keepdim=True)

        self.r1xr2 = torch.cross(r1, r2, dim=-1)
        self.r2xr3 = torch.cross(r2, r3, dim=-1)

        self.r2_norm = torch.linalg.vector_norm(r2,dim=-1,keepdim=True)
        self.r1_r2 = r1 * self.r2_norm

        self.phi = torch.atan2(
            (self.r1_r2 * self.r2xr3).sum(dim=-1),
            (self.r1xr2 * self.r2xr3).sum(dim=-1)
        ).unsqueeze(dim=-1)

        self.omega = r2
        self.omega_unit = F.normalize(r2, p=2, dim=-1)
        # self.omega_unit = F.normalize(coords[torsion_index[:,3],] - coords[torsion_index[:,0],], p=2, dim=-1)
    
    def geom(self):
        return self.phi
    
    def force_direction(self):
        return torch.stack([-self.omega_unit, self.omega_unit], dim=-1)

