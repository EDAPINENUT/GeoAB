#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import defaultdict
from copy import copy, deepcopy
import json
import math
import os
from typing import Dict, List, Optional, Tuple, Union
import requests

import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure as BStructure
from Bio.PDB.Model import Model as BModel
from Bio.PDB.Chain import Chain as BChain
from Bio.PDB.Residue import Residue as BResidue
from Bio.PDB.Atom import Atom as BAtom
from tqdm import tqdm
try:
    from .esm_utils import chain2esm
except:
    pass
from utils import print_log
PDB_PATH = os.path.join(os.path.dirname(os.path.split(__file__)[0]), 'all_data/pdb')

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix  # mat.dot(vec1) == vec2


# Gram-Schmidt Orthogonalization
def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum([np.dot(v, b) * b  for b in basis], axis=0)
        if (np.abs(w) > 1e-10).any():  
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)


class AminoAcid:
    def __init__(self, symbol, abrv, idx=0, side_chain_coord=None):
        self.symbol = symbol
        self.abrv = abrv
        self.idx = idx
        self.side_chain_coord = side_chain_coord

    def set_side_chain_coord(self, side_chain_coord):
        '''side_chain_coord:
        relative positions in the local coordinate with (CB-CA, N-CA, C-CA) (after gram schmidt)
        '''
        self.side_chain_coord = deepcopy(side_chain_coord)
        for atom in self.side_chain_coord:
            self.side_chain_coord[atom] = np.array(self.side_chain_coord[atom])

    def __str__(self):
        return f'{self.idx} {self.symbol} {self.abrv}'


class AminoAcidVocab:
    def __init__(self):
        self.PAD, self.SEP, self.UNK = '#', '/', '*'
        self.BOA, self.BOH, self.BOL = '&', '+', '-' # begin of antigen, heavy chain, light chain
        specials = [# special added (PAD)
                (self.PAD, 'PAD'), (self.UNK, 'UNK'),  # for RefineGNN, pad is mask
                (self.BOA, '<X>'), (self.BOH, '<H>'), (self.BOL, '<L>'), (self.SEP, '<E>')
            ]
        aas = [
                ('G', 'GLY'), ('A', 'ALA'), ('V', 'VAL'), ('L', 'LEU'),
                ('I', 'ILE'), ('F', 'PHE'), ('W', 'TRP'), ('Y', 'TYR'),
                ('D', 'ASP'), ('H', 'HIS'), ('N', 'ASN'), ('E', 'GLU'),
                ('K', 'LYS'), ('Q', 'GLN'), ('M', 'MET'), ('R', 'ARG'),
                ('S', 'SER'), ('T', 'THR'), ('C', 'CYS'), ('P', 'PRO'),
                ('U', 'SEC') # 21 aa for eukaryote
            ]
        _all = specials + aas
        self.amino_acids = [AminoAcid(symbol, abrv) for symbol, abrv in _all]
        self.symbol2idx, self.abrv2idx = {}, {}
        for i, aa in enumerate(self.amino_acids):
            self.symbol2idx[aa.symbol] = i
            self.abrv2idx[aa.abrv] = i
            aa.idx = i
        self.special_mask = [1 for _ in specials] + [0 for _ in aas]
        self.side_chain_loaded = False
    
    def load_side_chain_coord(self, path):
        self.side_chain_loaded = True
        return
        # not implemented
        with open(path, 'r') as fin:
            coord = json.load(fin)
        for symbol in coord:
            self.amino_acids[self.symbol_to_idx(symbol)].set_side_chain_coord(coord[symbol])
            # print(f'{symbol} side chain information loaded')
        print(f'{len(coord)} side chain information loaded')

    def abrv_to_symbol(self, abrv):
        idx = self.abrv_to_idx(abrv)
        return None if idx is None else self.amino_acids[idx].symbol

    def symbol_to_abrv(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return None if idx is None else self.amino_acids[idx].abrv

    def abrv_to_idx(self, abrv):
        abrv = abrv.upper()
        return self.abrv2idx.get(abrv, None)

    def symbol_to_idx(self, symbol):
        symbol = symbol.upper()
        return self.symbol2idx.get(symbol, None)
    
    def idx_to_symbol(self, idx):
        return self.amino_acids[idx].symbol

    def idx_to_abrv(self, idx):
        return self.amino_acids[idx].abrv

    def get_pad_idx(self):
        return self.symbol_to_idx(self.PAD)

    def get_unk_idx(self):
        return self.symbol_to_idx(self.UNK)
    
    def get_special_mask(self):
        return copy(self.special_mask)

    def get_side_chain_info(self, symbol):
        assert self.side_chain_loaded, 'Side chain information not loaded!'
        symbol = self.symbol_to_idx(symbol)
        coord = self.amino_acids[symbol].side_chain_coord
        return deepcopy(coord)

    def __len__(self):
        return len(self.symbol2idx)


VOCAB = AminoAcidVocab()
# SIDE_CHAIN_FILE = os.path.join(os.path.split(__file__)[0], 'side_chain_coords.json')
# if os.path.exists(SIDE_CHAIN_FILE):
#     VOCAB.load_side_chain_coord(SIDE_CHAIN_FILE)
# else:
#     print_log(f'Side chain coordination file not found at {SIDE_CHAIN_FILE}')


def format_aa_abrv(abrv):  # special cases
    if abrv == 'MSE':
        return 'MET' # substitue MSE with MET
    return abrv


class Residue:
    def __init__(self, symbol: str, coordinate: Tuple, _id: Tuple):
        self.symbol = symbol
        self.coordinate = coordinate
        self.id = _id  # (residue_number, insert_code)

    def get_symbol(self):
        return self.symbol

    def get_coord(self, atom_name):
        return copy(self.coordinate[atom_name])

    def get_coord_map(self) -> Dict[str, List]:
        return deepcopy(self.coordinate)

    def get_backbone_coord_map(self) -> Dict[str, List]:
        coord = { atom: self.coordinate[atom] for atom in self.coordinate if atom in ['CA', 'C', 'N', 'O'] }
        return coord

    def get_side_chain_coord_map(self) -> Dict[str, List]:
        coord = { atom: self.coordinate[atom] for atom in self.coordinate if atom not in ['CA', 'C', 'N', 'O'] }
        return coord

    def get_atom_names(self):
        return list(self.coordinate.keys())

    def get_id(self):
        return self.id

    def set_symbol(self, symbol):
        assert VOCAB.symbol_to_abrv(symbol) is not None, f'{symbol} is not an amino acid'
        self.symbol = symbol

    def set_coord(self, coord):
        self.coordinate = deepcopy(coord)

    def gen_side_chain(self, center=None):
        '''
        automatically generate side chain according to statistics
        '''
        coord = VOCAB.get_side_chain_info(self.symbol)
        if coord is None:  # do not have a side chain
            return
        n, ca = np.array(self.get_coord('N')), np.array(self.get_coord('CA'))
        if center is None:
            center = coord['center'] + ca
        else:
            center = np.array(center)
        coord.pop('center')

        center_ca, n_ca = center - ca, n - ca
        X = gram_schmidt([center_ca, n_ca, np.cross(center_ca, n_ca)])
        for atom in coord:
            self.coordinate[atom] = (coord[atom].dot(X) + ca).tolist()

    def dist_to(self, residue):  # measured by nearest atoms
        xa = np.array(list(self.get_coord_map().values()))
        xb = np.array(list(residue.get_coord_map().values()))
        if len(xa) == 0 or len(xb) == 0:
            return math.nan
        dist = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=-1)
        return np.min(dist)

    def to_bio(self):
        _id = (' ', self.id[0], self.id[1])
        residue = BResidue(_id, VOCAB.symbol_to_abrv(self.symbol), '    ')
        atom_map = self.coordinate
        for i, atom in enumerate(atom_map):
            fullname = ' ' + atom
            while len(fullname) < 4:
                fullname += ' '
            bio_atom = BAtom(
                name=atom,
                coord=np.array(atom_map[atom], dtype=np.float32),
                bfactor=0,
                occupancy=1.0,
                altloc=' ',
                fullname=fullname,
                serial_number=i,
                element=atom[0]  # not considering symbols with 2 chars (e.g. FE, MG)
            )
            residue.add(bio_atom)
        return residue


class Peptide:
    def __init__(self, _id, residues: List[Residue]):
        self.residues = residues
        self.seq = ''
        self.id = _id
        for residue in residues:
            self.seq += residue.get_symbol()

    def set_id(self, _id):
        self.id = _id

    def get_id(self):
        return self.id

    def get_seq(self):
        return self.seq

    def get_span(self, i, j):  # [i, j)
        i, j = max(i, 0), min(j, len(self.seq))
        if j <= i:
            residues = []
        else:
            residues = deepcopy(self.residues[i:j])
        return Peptide(self.id, residues)

    def get_residue(self, i):
        return deepcopy(self.residues[i])
    
    def get_ca_pos(self, i):
        return copy(self.residues[i].get_coord('CA'))

    def get_cb_pos(self, i):
        return copy(self.residues[i].get_coord('CB'))

    def set_residue_coord(self, i, coord):
        self.residues[i].set_coord(coord)

    def set_residue_translation(self, i, vec):
        coord = self.residues[i].get_coord_map()
        for atom in coord:
            ori_vec = coord[atom]
            coord[atom] = [a + b for a, b in zip(ori_vec, vec)]
        self.set_residue_coord(i, coord)

    def set_residue_symbol(self, i, symbol):
        self.residues[i].set_symbol(symbol)
        self.seq = self.seq[:i] + symbol + self.seq[i+1:]

    def set_residue(self, i, symbol, coord, center=None, gen_side_chain=False):
        self.set_residue_symbol(i, symbol)
        self.set_residue_coord(i, coord)
        if gen_side_chain:
            self.residues[i].gen_side_chain(center)

    def to_bio(self):
        chain = BChain(id=self.id)
        for residue in self.residues:
            chain.add(residue.to_bio())
        return chain

    def __len__(self):
        return len(self.seq)

    def __str__(self):
        return self.seq


class Protein:
    def __init__(self, pdb_id, peptides):
        self.pdb_id = pdb_id
        self.peptides = peptides

    @classmethod
    def from_pdb(cls, pdb_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('anonym', pdb_path)
        pdb_id = structure.header['idcode'].upper()
        peptides = {}
        for chain in structure.get_chains():
            _id = chain.get_id()
            residues = []
            has_non_residue = False
            for residue in chain:
                abrv = residue.get_resname()
                hetero_flag, res_number, insert_code = residue.get_id()
                if hetero_flag != ' ':
                    continue   # residue from glucose or water
                symbol = VOCAB.abrv_to_symbol(abrv)
                if symbol is None:
                    has_non_residue = True
                    print(f'has non residue: {abrv}')
                    break
                # filter Hs because not all data include them
                atoms = { atom.get_id(): atom.get_coord() for atom in residue if atom.element != 'H' }
                residues.append(Residue(
                    symbol, atoms, (res_number, insert_code)
                ))
            if has_non_residue or len(residues) == 0:  # not a peptide
                continue
            peptides[_id] = Peptide(_id, residues)
        return cls(pdb_id, peptides)

    def get_id(self):
        return self.pdb_id

    def num_chains(self):
        return len(self.peptides)

    def get_chain(self, name):
        if name in self.peptides:
            return deepcopy(self.peptides[name])
        else:
            return Peptide(name, [])

    def get_chain_names(self):
        return list(self.peptides.keys())

    def to_bio(self):
        structure = BStructure(id=self.pdb_id)
        model = BModel(id=0)
        for name in self.peptides:
            model.add(self.peptides[name].to_bio())
        structure.add(model)
        return structure

    def to_pdb(self, path):
        bio_structure = self.to_bio()
        io = PDBIO()
        io.set_structure(bio_structure)
        io.save(path)

    def __eq__(self, other):
        if not isinstance(other, Protein):
            raise TypeError('Cannot compare other type to Protein')
        for key in self.peptides:
            if key in other.peptides and self.peptides[key].seq == other.peptides[key].seq:
                continue
            else:
                return False
        return True

    def __str__(self):
        res = self.pdb_id + '\n'
        for seq_name in self.peptides:
            res += f'\t{seq_name}: {self.peptides[seq_name]}\n'
        return res


def continuous_segments(indexes):
    segs, indexes = [], sorted(list(set(indexes)))
    return indexes
    print(len(indexes))
    cur_seg = None
    for i in indexes:
        if cur_seg is None:
            cur_seg = [i, i]
        elif i <= cur_seg[1]:
            cur_seg[1] = i
        else:  # finish a segment
            if cur_seg[0] != cur_seg[1]:  # drop single residue
                segs.append(cur_seg)
            cur_seg = [i, i]
    if cur_seg is not None and cur_seg[0] != cur_seg[1]:
        segs.append(cur_seg)
    return segs


class AAComplex(Protein):  # Antibody-Antigen complex

    # threshold = 6.6  # from "Characterization of Protein-Protein Interfaces"
    num_interface_residue = 48  # from PNAS, Jian Peng

    def __init__(self, pdb_id: str, peptides: Dict[str, Peptide], heavy_chain: str, 
                 light_chain: str, antigen_chains: List[str], cdr_type: str, use_esm: bool=False, 
                 numbering: str='imgt', cdr_pos: Optional[Dict[str, Tuple]]=None, 
                 skip_cal_interface=False):
        '''
            heavy_chain: the id of heavy chain
            light_chain: the id of light chain
            antigen_chains: the list of ids of antigen chains
            numbering: currently only support IMGT
            self.cdr_pos: dict with keys like CDR-H3 and values like (40, 45) indicating [begin, end]
        '''
        self.heavy_chain = heavy_chain
        self.light_chain = light_chain
        self.antigen_chains = copy(antigen_chains)
        self.use_esm = use_esm
        self.cdr_type = cdr_type
    
        # antibody information
        if cdr_pos is None:
            selected_peptides, self.cdr_pos = self._extract_antibody_info(peptides, numbering, cdr_type)
        else:
            selected_peptides, self.cdr_pos = {}, deepcopy(cdr_pos)
            for chain_name in [heavy_chain, light_chain]:
                if chain_name in peptides:
                    selected_peptides[chain_name] = peptides[chain_name]

        if use_esm:
            self.esm_h = chain2esm(heavy_chain, selected_peptides[heavy_chain].mask_seq)

            self.esm_l = chain2esm(light_chain, selected_peptides[light_chain].mask_seq)
        else:
            self.esm_h = None
            self.esm_l = None

        # antigen information
        self.esm_ag = {} # perhaps multiple antigens
        for chain_name in antigen_chains:

            assert chain_name not in selected_peptides, f'Antigen chain {chain_name} is antibody itself!'
            
            selected_peptides[chain_name] = peptides[chain_name]

            if use_esm:
                self.esm_ag[chain_name] = chain2esm(chain_name, selected_peptides[chain_name].seq)
            else:
                self.esm_ag[chain_name] = None

        super().__init__(pdb_id, selected_peptides)

        if not skip_cal_interface:
            self._cal_interface(self.esm_ag)

    def _extract_antibody_info(self, peptides, numbering, cdr_type):
        # calculating cdr pos according to number scheme (type_mapping and conserved residues)
        if numbering.lower() == 'imgt':
            type_mapping = {}  # - for non-Fv region, 0 for framework, 1/2/3 for cdr1/2/3
            for i in list(range(1, 27)) + list(range(39, 56)) + list(range(66, 105)) + list(range(118, 130)):
                type_mapping[i] = '0'
            for i in range(27, 39):     # cdr1
                type_mapping[i] = '1'
            for i in range(56, 66):     # cdr2
                type_mapping[i] = '2'
            for i in range(105, 118):   # cdr3
                type_mapping[i] = '3'
            conserved = {
                23: ['CYS'],
                41: ['TRP'],
                104: ['CYS'],
                # 118: ['PHE', 'TRP']
            }
        else:
            raise NotImplementedError(f'Numbering scheme {numbering} not implemented')

        selected_peptides, cdr_pos = {}, {}
        for c, chain_name in zip(['H', 'L'], [self.heavy_chain, self.light_chain]):
            chain = None
            for _name in [chain_name, chain_name.upper()]:
                # Note: possbly two chains are different segments of a same chain
                if _name in peptides:
                    chain = peptides[_name]
                    break
            if chain is None:
                continue           
            res_type = ''
            for i in range(len(chain)):
                residue = chain.get_residue(i)
                residue_number = residue.get_id()[0]
                if residue_number in type_mapping:
                    res_type += type_mapping[residue_number]
                    if residue_number in conserved:
                        hit, symbol = False, residue.get_symbol()
                        for conserved_residue in conserved[residue_number]:
                            if symbol == VOCAB.abrv_to_symbol(conserved_residue):
                                hit = True
                                break
                        assert hit, f'Not {conserved[residue_number]} at {residue_number}'
                else:
                    res_type += '-'
            start, end = res_type.index('0'), res_type.rindex('0')
            for cdr in ['1', '2', '3']:
                cdr_start, cdr_end = res_type.find(cdr), res_type.rfind(cdr)
                if cdr_start == -1:
                    raise ValueError(f'cdr {cdr} not found, residue type: {res_type}')
                start, end = min(start, cdr_start), max(end, cdr_end)
                cdr_pos[f'CDR-{c}{cdr}'] = (cdr_start, cdr_end)
            for cdr in ['1', '2', '3']:
                cdr = f'CDR-{c}{cdr}'
                cdr_start, cdr_end = cdr_pos[cdr]
                cdr_pos[cdr] = (cdr_start - start, cdr_end - start)
            chain = chain.get_span(start, end + 1)  # the length may exceed 130 because of inserted amino acids
            chain.set_id(chain_name)
            selected_peptides[chain_name] = chain

            #--------------------------------------------------------------
            # Mask cdr according to cdr_type, <mask>
            #--------------------------------------------------------------
            CDRstart, CDRend = cdr_pos['CDR-{}{}'.format(c, cdr_type)] 

            if c == 'H':
                maskseq = selected_peptides[chain_name].seq[: CDRstart] + \
                          ''.join(['<mask>'] *(CDRend-CDRstart+1)) + \
                          selected_peptides[chain_name].seq[CDRend+1:]

                selected_peptides[chain_name].mask_seq = maskseq 
            else:
                selected_peptides[chain_name].mask_seq = selected_peptides[chain_name].seq 

        return selected_peptides, cdr_pos

    def _cal_interface(self, esm_ag=None):
        antigen_chains = self.get_antigen_chains()
        antigen_names = [peptide.get_id() for peptide in antigen_chains]
        antibody_chains = [self.get_heavy_chain(), self.get_light_chain()]
        antibody_chains = [chain for chain in antibody_chains if chain is not None]
        antibody_names = [chain.get_id() for chain in antibody_chains]
        coord = {}
        for name, chain in zip(antigen_names + antibody_names, antigen_chains + antibody_chains):
            coord[name] = {'x': [], 'i': [], 'interface': []}
            for i in range(len(chain)):
                try:
                    coord[name]['x'].append(chain.get_ca_pos(i))
                except KeyError:
                    print_log(f'{self.pdb_id}, chain {chain.get_id()}, residue at {chain.get_residue(i).get_id()} has no ca coordination, fill with other atoms')
                    coord_map = chain.get_residue(i).get_coord_map()
                    atom = list(coord_map.keys())[0]
                    coord[name]['x'].append(coord_map[atom])
                # coord[name]['i'].append(i)
                # residue = chain.get_residue(i)
                # for atom in residue.get_atom_names():
                #     x = residue.get_coord(atom)
                #     coord[name]['x'].append(x)
                #     coord[name]['i'].append(i)

        for name in coord:
            coord[name]['x'] = np.array(coord[name]['x'])
            # coord[name]['i'] = np.array(coord[name]['i'])

        # calculate distance
        min_dists = { name: [] for name in antigen_names }
        for name in antigen_names:
            xa = coord[name]['x']
            for ab_name in antibody_names:
                xb = coord[ab_name]['x']
                if len(xb) == 0:  # for single-domain antibodies
                    continue
                dist = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=-1)
                min_dist = np.min(dist, axis=1) # [num_ag_residue]
                min_dists[name].append(min_dist)
                # _is, _js = np.nonzero(dist <= self.threshold)
                # _is, _js = coord[name]['i'][_is], coord[ab_name]['i'][_js]
                # for n, indexes in zip([name, ab_name], [_is, _js]):
                #     # if indexes.shape[0] == 0:
                #     #     break
                #     # print(indexes)
                #     # start, end = np.min(indexes), np.max(indexes)
                #     # first put all index into it
                #     coord[n]['interface'].extend(set(indexes))
                #     # coord[n]['interface'][0] = min(coord[n]['interface'][0], start)
                #     # coord[n]['interface'][1] = max(coord[n]['interface'][1], end)
        dists, ids = [], []
        for name in antigen_names:
            if len(min_dists[name]) == 2:
                min_dists[name] = np.minimum(*min_dists[name])
            elif len(min_dists[name]) == 1:
                min_dists[name] = min_dists[name][0]
            else:
                raise ValueError('Number of chain in ab exceed 2')
            dists.extend(min_dists[name])
            for i in range(len(min_dists[name])):
                ids.append((name, i))
        dists = np.array(dists)
        topk = min(len(dists), self.num_interface_residue)
        ind = np.argpartition(-dists, -topk)[-topk:]
        self.antigen_interface = {}
        self.esm_interface = {}
        for i in ind:
            name, residue_idx = ids[i]
            if name not in self.antigen_interface:
                self.antigen_interface[name] = []
                self.esm_interface[name] = []
            self.antigen_interface[name].append(residue_idx)
            if self.use_esm:
                self.esm_interface[name].append(esm_ag[name][residue_idx])

    def get_heavy_chain(self, interface_only=False) -> Union[Peptide, List[Peptide]]:
        chain = self.get_chain(self.heavy_chain)
        if hasattr(self, 'esm_h'):
            if self.esm_h is not None: 
                chain.esm = self.esm_h
        if not len(chain):
            return chain
        if interface_only:
            raise NotImplementedError('get heavy chain interface not implemented')
            spans = self.antibody_interface[self.heavy_chain]
            chain = Peptide(chain.get_id(), [chain.get_residue(i) for i in spans])
        return chain

    def get_light_chain(self, interface_only=False) -> Union[Peptide, List[Peptide]]:
        chain = self.get_chain(self.light_chain)
        if hasattr(self, 'esm_h'):
            if self.esm_h is not None: 
                chain.esm = self.esm_h
        if not len(chain):
            return chain
        if interface_only:
            raise NotImplementedError('get light chain interface not implemented')
            spans = self.antibody_interface[self.light_chain]
            chain = [chain.get_span(start, end + 1) for start, end in spans]
        return chain

    def get_antigen_chains(self, interface_only=False, cdr=None) -> List[Peptide]:
        chains = []
         # H/L + 1/2/3, None for the whole antibody
        for name in self.antigen_chains:
            chain = self.get_chain(name)
            if hasattr(self, 'esm_ag'):
                if self.esm_ag is not None:
                    chain.esm = self.esm_ag[name]    
            chains.append(self.get_chain(name))

        if interface_only:
            new_chains = []
            cdr = None if cdr is None else f'CDR-{cdr.upper()}'
            for chain in chains:
                if cdr is None:
                    spans = self.antigen_interface.get(chain.get_id(), [])
                    esms = self.esm_interface.get(chain.get_id(), []) if hasattr(self, 'esm_interface') else []
                    esms = np.stack(esms, axis=0) if len(esms) else None
                else:
                    name = f'{chain.get_id()}-{cdr}'
                    spans = self.antigen_cdr_interface.get(name, [])
                    esms = self.esm_cdr_interface.get(name, [])
                    esms = np.stack(esms, axis=0) if len(esms) else None
                new_chain = Peptide(chain.get_id(), [chain.get_residue(i) for i in spans])
                if esms is not None:
                    new_chain.esm = esms  
                new_chains.append(new_chain)
            chains = [chain for chain in new_chains if len(chain)]
        return chains

    def get_cdr_pos(self, cdr='H3'):  # H/L + 1/2/3, return [begin, end] position
        cdr = f'CDR-{cdr}'.upper()
        if cdr in self.cdr_pos:
            return self.cdr_pos[cdr]
        else:
            return (-1, -1)

    def get_cdr(self, cdr='H3'):
        cdr = cdr.upper()
        pos = self.get_cdr_pos(cdr)
        chain = self.get_heavy_chain() if 'H' in cdr else self.get_light_chain()
        return chain.get_span(pos[0], pos[1] + 1)

    def __str__(self):
        pdb_info = f'PDB ID: {self.pdb_id}'
        antibody_info = f'Antibody H-{self.heavy_chain} ({len(self.get_heavy_chain())}), ' + \
                        f'L-{self.light_chain} ({len(self.get_light_chain())})'
        antigen_info = f'Antigen Chains: {[(ag, len(self.get_chain(ag))) for ag in self.antigen_chains]}'
        cdr_info = f'CDRs: \n'
        for name in self.cdr_pos:
            chain = self.get_heavy_chain() if 'H' in name else self.get_light_chain()
            start, end = self.cdr_pos[name]
            cdr_info += f'\t{name}: [{start}, {end}], {chain.seq[start:end + 1]}\n'
        sep = '\n' + '=' * 20 + '\n'
        return sep + pdb_info + '\n' + antibody_info + '\n' + cdr_info + '\n' + antigen_info + sep


def construct_side_chain_coord(pdb_dir, out_path):
    side_chain_coord = {}
    # line = 0
    for f in tqdm(os.listdir(pdb_dir)):
        # line += 1
        # if line > 20:
        #     break
        try:
            protein = Protein.from_pdb(os.path.join(pdb_dir, f))
        except Exception as e:
            print(f'{e}, skip')
            continue
        for c in protein.get_chain_names():
            chain = protein.get_chain(c)
            for i in range(len(chain)):
                residue = chain.get_residue(i)
                atom_coord = residue.get_coord_map()
                try:
                    ca, n = atom_coord['CA'], atom_coord['N']
                    c = atom_coord['C']
                except KeyError:  # coordinates are missing
                    continue
                side_chain_atom_coord = residue.get_side_chain_coord_map()
                if len(side_chain_atom_coord) < 2:  # only one CB or no heavy atoms (GLY)
                    continue
                center = [side_chain_atom_coord[atom] for atom in side_chain_atom_coord]
                center = np.mean(center, axis=0)
                center_ca, n_ca = np.array([center, n]) - np.array([ca, ca])
                X, ca = gram_schmidt([center_ca, n_ca, np.cross(center_ca, n_ca)]), np.array(ca)
                symbol = residue.get_symbol()
                if symbol not in side_chain_coord:
                    side_chain_coord[symbol] = defaultdict(list)
                side_chain_coord[symbol]['center'].append(center_ca)
                for atom in side_chain_atom_coord:
                    x = np.array(atom_coord[atom]) - ca
                    assert len(x) == 3, x
                    x = X.dot(x)  # coordinate in the local frame
                    assert len(x) == 3, f'local: {x}, {X}, {center_ca}, {n_ca}'
                    side_chain_coord[symbol][atom].append(x)
    for symbol in side_chain_coord:
        print(f'{symbol}:')
        atom_coords = {}
        for atom in side_chain_coord[symbol]:
            coord = np.mean(side_chain_coord[symbol][atom], axis=0).tolist()
            std = np.std(side_chain_coord[symbol][atom], axis=0)
            print(f'\t{atom}: {coord}, {std}')
            atom_coords[atom] = coord
        side_chain_coord[symbol] = atom_coords
    with open(out_path, 'w') as fout:
        json.dump(side_chain_coord, fout)


def fetch_from_pdb(identifier):
    # example identifier: 1FBI
    url = 'https://data.rcsb.org/rest/v1/core/entry/' + identifier
    res = requests.get(url)
    if res.status_code != 200:
        return None
    url = f'https://files.rcsb.org/download/{identifier}.pdb'
    text = requests.get(url)
    data = res.json()
    data['pdb'] = text.text
    return data


class AgAbComplex:

    num_interface_residues = 48  # from PNAS (view as epitope)

    def __init__(self, antigen: Protein, antibody: Protein, heavy_chain: str, light_chain: str,
                 numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False) -> None:
        self.heavy_chain = heavy_chain
        self.light_chain = light_chain
        self.numbering = numbering

        self.antigen = antigen
        if skip_validity_check:
            self.antibody, self.cdr_pos = antibody, None
        else:
            self.antibody, self.cdr_pos = self._extract_antibody_info(antibody, numbering)
        self.pdb_id = antigen.get_id()

        if skip_epitope_cal:
            self.epitope = None
        else:
            self.epitope = self._cal_epitope()
    
    @classmethod
    def from_pdb(cls, pdb_path: str, heavy_chain: str, light_chain: str, antigen_chains: List[str],
                 numbering: str='imgt', skip_epitope_cal=False, skip_validity_check=False):
        protein = Protein.from_pdb(pdb_path)
        pdb_id = protein.get_id()
        ab_peptides = {
            heavy_chain: protein.get_chain(heavy_chain),
            light_chain: protein.get_chain(light_chain)
        }
        ag_peptides = { chain: protein.get_chain(chain) for chain in antigen_chains if protein.get_chain(chain) is not None }
        for chain in antigen_chains:
            assert chain in ag_peptides, f'Antigen chain {chain} has something wrong!'

        antigen = Protein(pdb_id, ag_peptides)
        antibody = Protein(pdb_id, ab_peptides)

        return cls(antigen, antibody, heavy_chain, light_chain, numbering, skip_epitope_cal, skip_validity_check)

    def _extract_antibody_info(self, antibody: Protein, numbering: str):
        # calculating cdr pos according to number scheme (type_mapping and conserved residues)
        numbering = numbering.lower()
        if numbering == 'imgt':
            _scheme = IMGT
        elif numbering.lower() == 'chothia':
            _scheme = Chothia
            # for i in list(range(1, 27)) + list(range(39, 56)) + list(range(66, 105)) + list(range(118, 130)):
            #     type_mapping[i] = '0'
            # for i in range(27, 39):     # cdr1
            #     type_mapping[i] = '1'
            # for i in range(56, 66):     # cdr2
            #     type_mapping[i] = '2'
            # for i in range(105, 118):   # cdr3
            #     type_mapping[i] = '3'
            # conserved = {
            #     23: ['CYS'],
            #     41: ['TRP'],
            #     104: ['CYS'],
            #     # 118: ['PHE', 'TRP']
            # }
        else:
            raise NotImplementedError(f'Numbering scheme {numbering} not implemented')

        # get cdr/frame denotes
        h_type_mapping, l_type_mapping = {}, {}  # - for non-Fv region, 0 for framework, 1/2/3 for cdr1/2/3

        for lo, hi in [_scheme.HFR1, _scheme.HFR2, _scheme.HFR3, _scheme.HFR4]:
            for i in range(lo, hi + 1):
                h_type_mapping[i] = '0'
        for cdr, (lo, hi) in zip(['1', '2', '3'], [_scheme.H1, _scheme.H2, _scheme.H3]):
            for i in range(lo, hi + 1):
                h_type_mapping[i] = cdr
        h_conserved = _scheme.Hconserve

        for lo, hi in [_scheme.LFR1, _scheme.LFR2, _scheme.LFR3, _scheme.LFR4]:
            for i in range(lo, hi + 1):
                l_type_mapping[i] = '0'
        for cdr, (lo, hi) in zip(['1', '2', '3'], [_scheme.L1, _scheme.L2, _scheme.L3]):
            for i in range(lo, hi + 1):
                l_type_mapping[i] = cdr
        l_conserved = _scheme.Lconserve

        # get variable domain and cdr positions
        selected_peptides, cdr_pos = {}, {}
        for c, chain_name in zip(['H', 'L'], [self.heavy_chain, self.light_chain]):
            chain = antibody.get_chain(chain_name)
            # Note: possbly two chains are different segments of a same chain
            assert chain is not None, f'Chain {chain_name} not found in the antibody'
            type_mapping = h_type_mapping if c == 'H' else l_type_mapping
            conserved = h_conserved if c == 'H' else l_conserved
            res_type = ''
            for i in range(len(chain)):
                residue = chain.get_residue(i)
                residue_number = residue.get_id()[0]
                if residue_number in type_mapping:
                    res_type += type_mapping[residue_number]
                    if residue_number in conserved:
                        hit, symbol = False, residue.get_symbol()
                        for conserved_residue in conserved[residue_number]:
                            if symbol == VOCAB.abrv_to_symbol(conserved_residue):
                                hit = True
                                break
                        assert hit, f'Not {conserved[residue_number]} at {residue_number}'
                else:
                    res_type += '-'
            if '0' not in res_type:
                print(self.heavy_chain, self.light_chain, antibody.pdb_id, res_type)
            start, end = res_type.index('0'), res_type.rindex('0')
            for cdr in ['1', '2', '3']:
                cdr_start, cdr_end = res_type.find(cdr), res_type.rfind(cdr)
                assert cdr_start != -1, f'cdr {c}{cdr} not found, residue type: {res_type}'
                start, end = min(start, cdr_start), max(end, cdr_end)
                cdr_pos[f'CDR-{c}{cdr}'] = (cdr_start, cdr_end)
            for cdr in ['1', '2', '3']:
                cdr = f'CDR-{c}{cdr}'
                cdr_start, cdr_end = cdr_pos[cdr]
                cdr_pos[cdr] = (cdr_start - start, cdr_end - start)
            chain = chain.get_span(start, end + 1)  # the length may exceed 130 because of inserted amino acids
            chain.set_id(chain_name)
            selected_peptides[chain_name] = chain

        antibody = Protein(antibody.get_id(), selected_peptides)

        return antibody, cdr_pos

    def _cal_epitope(self):
        ag_rids, ag_xs, ab_xs = [], [], []
        ag_mask, ab_mask = [], []
        cdrh3 = self.get_cdr('H3')
        for _type, protein in zip(['ag', 'ab'], [self.antigen, [('A', cdrh3)]]):
            is_ag = _type == 'ag'
            rids = []
            if is_ag: 
                xs, masks = ag_xs, ag_mask
            else:
                xs, masks = ab_xs, ab_mask
            for chain_name, chain in protein:
                for i, residue in enumerate(chain):
                    bb_coord = residue.get_backbone_coord_map()
                    sc_coord = residue.get_sidechain_coord_map()
                    coord = {}
                    coord.update(bb_coord)
                    coord.update(sc_coord)
                    num_pad = VOCAB.MAX_ATOM_NUMBER - len(coord)
                    x = [coord[key] for key in coord] + [[0, 0, 0] for _ in range(num_pad)]
                    mask = [1 for _ in coord] + [0 for _ in range(num_pad)]
                    rids.append((chain_name, i))
                    xs.append(x)
                    masks.append(mask)
            if is_ag:
                ag_rids = rids
        assert len(ag_xs) != 0, 'No antigen structure!'
        # calculate distance
        ag_xs, ab_xs = np.array(ag_xs), np.array(ab_xs)  # [Nag/ab, M, 3], M == MAX_ATOM_NUM
        ag_mask, ab_mask = np.array(ag_mask).astype('bool'), np.array(ab_mask).astype('bool')  # [Nag/ab, M]
        dist = np.linalg.norm(ag_xs[:, None] - ab_xs[None, :], axis=-1)  # [Nag, Nab, M]
        dist = dist + np.logical_not(ag_mask[:, None] * ab_mask[None, :]) * 1e6  # [Nag, Nab, M]
        min_dists = np.min(np.min(dist, axis=-1), axis=-1)  # [ag_len]
        topk = min(len(min_dists), self.num_interface_residues)
        ind = np.argpartition(-min_dists, -topk)[-topk:]
        epitope = []
        for idx in ind:
            chain_name, i = ag_rids[idx]
            residue = self.antigen.peptides[chain_name].get_residue(i)
            epitope.append((residue, chain_name, i))
        return epitope

    def get_id(self) -> str:
        return self.antibody.pdb_id

    def get_antigen(self) -> Protein:
        return deepcopy(self.antigen)

    def get_epitope(self, cdrh3_pos=None) -> List[Tuple[Residue, str, int]]:
        if cdrh3_pos is not None:
            backup = self.cdr_pos
            self.cdr_pos = {'CDR-H3': [cdrh3_pos[0], cdrh3_pos[1]]}
            epitope = self._cal_epitope()
            self.cdr_pos = backup
            return deepcopy(epitope)
        if self.epitope is None:
            self.epitope = self._cal_epitope()
        return deepcopy(self.epitope)

    def get_interacting_residues(self, dist_cutoff=5) -> Tuple[List[int], List[int]]:
        ag_rids, ag_xs, ab_xs = [], [], []
        for chain_name in self.antigen.get_chain_names():
            chain = self.antigen.get_chain(chain_name)
            for i in range(len(chain)):
                try:
                    x = chain.get_ca_pos(i)
                except KeyError:  # CA position is missing
                    continue
                ag_rids.append((chain_name, i))
                ag_xs.append(x)
        for chain_name in self.antibody.get_chain_names():
            chain = self.antibody.get_chain(chain_name)
            for i in range(len(chain)):
                try:
                    x = chain.get_ca_pos(i)
                except KeyError:
                    continue
                ab_xs.append(x)
        assert len(ag_xs) != 0, 'No antigen structure!'
        # calculate distance
        ag_xs, ab_xs = np.array(ag_xs), np.array(ab_xs)
        dist = np.linalg.norm(ag_xs[:, None, :] - ab_xs[None, :, :], axis=-1)
        min_dists = np.min(dist, axis=1)  # [ag_len]
        topk = min(len(min_dists), self.num_interface_residues)
        ind = np.argpartition(-min_dists, -topk)[-topk:]
        epitope = []
        for idx in ind:
            chain_name, i = ag_rids[idx]
            residue = self.antigen.peptides[chain_name].get_residue(i)
            epitope.append((residue, chain_name, i))
        return

    def get_heavy_chain(self) -> Peptide:
        return self.antibody.get_chain(self.heavy_chain)

    def get_light_chain(self) -> Peptide:
        return self.antibody.get_chain(self.light_chain)

    def get_framework(self, fr):  # H/L + FR + 1/2/3/4
        seg_id = int(fr[-1])
        chain = self.get_heavy_chain() if fr[0] == 'H' else self.get_light_chain()
        begin, end = -1, -1
        if seg_id == 1:
            begin, end = 0, self.get_cdr_pos(fr[0] + str(seg_id))[0]
        elif seg_id == 4:
            begin, end = self.get_cdr_pos(fr[0] + '3')[-1] + 1, len(chain)
        else:
            begin = self.get_cdr_pos(fr[0] + str(seg_id - 1))[-1] + 1
            end = self.get_cdr_pos(fr[0] + str(seg_id))[0]
        return chain.get_span(begin, end)

    def get_cdr_pos(self, cdr='H3'):  # H/L + 1/2/3, return [begin, end] position
        cdr = f'CDR-{cdr}'.upper()
        if cdr in self.cdr_pos:
            return self.cdr_pos[cdr]
        else:
            return None

    def get_cdr(self, cdr='H3'):
        cdr = cdr.upper()
        pos = self.get_cdr_pos(cdr)
        if pos is None:
            return None
        chain = self.get_heavy_chain() if 'H' in cdr else self.get_light_chain()
        return chain.get_span(pos[0], pos[1] + 1)

    def to_pdb(self, path, atoms=None):
        peptides = {}
        for name in self.antigen.get_chain_names():
            peptides[name] = self.antigen.get_chain(name)
        for name in self.antibody.get_chain_names():
            peptides[name] = self.antibody.get_chain(name)
        protein = Protein(self.get_id(), peptides)
        protein.to_pdb(path, atoms)
    
    def __str__(self):
        pdb_info = f'PDB ID: {self.pdb_id}'
        antibody_info = f'Antibody H-{self.heavy_chain} ({len(self.get_heavy_chain())}), ' + \
                        f'L-{self.light_chain} ({len(self.get_light_chain())})'
        antigen_info = f'Antigen Chains: {[(ag, len(self.antigen.get_chain(ag))) for ag in self.antigen.get_chain_names()]}'
        cdr_info = f'CDRs: \n'
        for name in self.cdr_pos:
            chain = self.get_heavy_chain() if 'H' in name else self.get_light_chain()
            start, end = self.cdr_pos[name]
            cdr_info += f'\t{name}: [{start}, {end}], {chain.seq[start:end + 1]}\n'
        epitope_info = f'Epitope: \n'
        residue_map = {}
        for _, chain_name, i in self.get_epitope():
            if chain_name not in residue_map:
                residue_map[chain_name] = []
            residue_map[chain_name].append(i)
        for chain_name in residue_map:
            epitope_info += f'\t{chain_name}: {sorted(residue_map[chain_name])}\n'

        sep = '\n' + '=' * 20 + '\n'
        return sep + pdb_info + '\n' + antibody_info + '\n' + cdr_info + '\n' + antigen_info + '\n' + epitope_info + sep



if __name__ == '__main__':
    import sys
    construct_side_chain_coord(sys.argv[1], sys.argv[2])