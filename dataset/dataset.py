import os
import json
import torch
import pickle
import functools
import numpy as np
from tqdm import tqdm
from typing import List
from copy import deepcopy

from utils import print_log
from data.pdb_utils import AAComplex, Protein, VOCAB, PDB_PATH, AgAbComplex

class PairedMutDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, save_dir=None, cdr=None, paratope='H3', full_antigen=False, num_entry_per_file=-1, random=False):
        '''
        file_path: path to the dataset
        save_dir: directory to save the processed data
        cdr: which cdr to generate (L1/2/3, H1/2/3) (can be list), None for all including framework
        paratope: which cdr to use as paratope (L1/2/3, H1/2/3) (can be list)
        full_antigen: whether to use the full antigen information
        num_entry_per_file: number of entries in a single file. -1 to save all data into one file 
                            (In-memory dataset)
        '''
        super().__init__()
        self.cdr = cdr
        self.paratope = paratope
        self.full_antigen = full_antigen
        self.mean_context = False
        if save_dir is None:
            if not os.path.isdir(file_path):
                save_dir = os.path.split(file_path)[0]
            else:
                save_dir = file_path
            prefix = os.path.split(file_path)[1]
            if '.' in prefix:
                prefix = prefix.split('.')[0]
            save_dir = os.path.join(save_dir, f'{prefix}_processed')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metainfo_file = os.path.join(save_dir, '_metainfo')
        self.data: List[AgAbComplex] = []  # list of ABComplex

        # try loading preprocessed files
        need_process = False
        try:
            with open(metainfo_file, 'r') as fin:
                metainfo = json.load(fin)
                self.num_entry = metainfo['num_entry']
                self.file_names = metainfo['file_names']
                self.file_num_entries = metainfo['file_num_entries']
        except FileNotFoundError:
            print_log('No meta-info file found, start processing', level='INFO')
            need_process = True
        except Exception as e:
            print_log(f'Faild to load file {metainfo_file}, error: {e}', level='WARN')
            need_process = True

        if need_process:
            # preprocess
            self.file_names, self.file_num_entries = [], []
            self.preprocess(file_path, save_dir, num_entry_per_file)
            self.num_entry = sum(self.file_num_entries)

            metainfo = {
                'num_entry': self.num_entry,
                'file_names': self.file_names,
                'file_num_entries': self.file_num_entries
            }
            with open(metainfo_file, 'w') as fout:
                json.dump(metainfo, fout)

        self.random = random
        self.cur_file_idx, self.cur_idx_range = 0, (0, self.file_num_entries[0])  # left close, right open
        self._load_part()

        # user defined variables
        self.idx_mapping = [i for i in range(self.num_entry)]
        self.mode = '111'  # H/L/Antigen, 1 for include, 0 for exclude

    def _save_part(self, save_dir, num_entry):
        file_name = os.path.join(save_dir, f'part_{len(self.file_names)}.pkl')
        print_log(f'Saving {file_name} ...')
        file_name = os.path.abspath(file_name)
        if num_entry == -1:
            end = len(self.data)
        else:
            end = min(num_entry, len(self.data))
        with open(file_name, 'wb') as fout:
            pickle.dump(self.data[:end], fout)
        self.file_names.append(file_name)
        self.file_num_entries.append(end)
        self.data = self.data[end:]

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        print_log(f'Loading preprocessed file {f}, {self.cur_file_idx + 1}/{len(self.file_names)}')
        with open(f, 'rb') as fin:
            del self.data
            self.data = pickle.load(fin)
        self.access_idx = [i for i in range(len(self.data))]
        if self.random:
            np.random.shuffle(self.access_idx)

    def _check_load_part(self, idx):
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        idx = self.access_idx[idx - self.cur_idx_range[0]]
        return idx
     
    def __len__(self):
        return self.num_entry

    ########### load data from file_path and add to self.data ##########
    def preprocess(self, file_path, save_dir, num_entry_per_file):
        '''
        Load data from file_path and add processed data entries to self.data.
        Remember to call self._save_data(num_entry_per_file) to control the number
        of items in self.data (this function will save the first num_entry_per_file
        data and release them from self.data) e.g. call it when len(self.data) reaches
        num_entry_per_file.
        '''
        with open(file_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        # line_id = 0
        for line in tqdm(lines):
            # if line_id < 206:
            #     line_id += 1
            #     continue
            item = json.loads(line)
            try:
                cplx = AgAbComplex.from_pdb(
                    item['pdb_data_path'], item['heavy_chain'], item['light_chain'],
                    item['antigen_chains'])
            except AssertionError as e:
                print_log(e, level='ERROR')
                print_log(f'parse {item["pdb"]} pdb failed, skip', level='ERROR')
                continue

            self.data.append(cplx)
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

    ########## override get item ##########
    def __getitem__(self, idx):
        '''
        an example of the returned data
        {
            'X': [n, n_channel, 3],
            'S': [n],
            'cmask': [n],
            'smask': [n],
            'paratope_mask': [n],
            'xloss_mask': [n, n_channel],
            'template': [n, n_channel, 3]
        }
        '''
        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)
        item = self.data[idx]

        # antigen
        ag_residues = []

        if self.full_antigen:
            # get antigen residues
            ag = item.get_antigen()
            for chain in ag.get_chain_names():
                chain = ag.get_chain(chain)
                for i in range(len(chain)):
                    residue = chain.get_residue(i)
                    ag_residues.append(residue)
        else:
            # get antigen residues (epitope only)
            for residue, chain, i in item.get_epitope():
                ag_residues.append(residue)
    
        # generate antigen data
        ag_data = _generate_chain_data(ag_residues, VOCAB.BOA)

        hc, lc = item.get_heavy_chain(), item.get_light_chain()
        hc_residues, lc_residues = [], []

        # generate heavy chain data
        for i in range(len(hc)):
            hc_residues.append(hc.get_residue(i))
        hc_data = _generate_chain_data(hc_residues, VOCAB.BOH)

        # generate light chain data
        for i in range(len(lc)):
            lc_residues.append(lc.get_residue(i))
        lc_data = _generate_chain_data(lc_residues, VOCAB.BOL)

        data = { key: np.concatenate([ag_data[key], hc_data[key], lc_data[key]], axis=0) \
                 for key in hc_data}

        # smask (sequence) and cmask (coordinates): 0 for fixed, 1 for generate
        # not generate coordinates of global node and antigen 
        cmask = [0 for _ in ag_data['S']] + [0] + [1 for _ in hc_data['S'][1:]] + [0] + [1 for _ in lc_data['S'][1:]]
        # according to the setting of cdr
        if self.cdr is None:
            smask = cmask
        else:
            smask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']) + len(lc_data['S']))]
            cdrs = [self.cdr] if type(self.cdr) == str else self.cdr
            for cdr in cdrs:
                cdr_range = item.get_cdr_pos(cdr)
                offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
                for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                    smask[idx] = 1
        
        data['cmask'], data['smask'] = cmask, smask ######################## NOTE: smask == cmask for the same context

        paratope_mask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']) + len(lc_data['S']))]
        paratope = [self.paratope] if type(self.paratope) == str else self.paratope
        for cdr in paratope:
            cdr_range = item.get_cdr_pos(cdr)
            offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
            for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                paratope_mask[idx] = 1
        data['paratope_mask'] = paratope_mask

        return data

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'S', 'smask', 'cmask', 'paratope_mask', 'residue_pos', 'template', 'xloss_mask']
        types = [torch.float, torch.long, torch.bool, torch.bool, torch.bool, torch.long, torch.float, torch.bool]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        lengths = [len(item['S']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        return res





# use this class to splice the dataset and maintain only one part of it in RAM
class EquiAACDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, cdr_type='3', save_dir=None, use_esm=False, num_entry_per_file=-1, random=False, ctx_cutoff=8.0, interface_cutoff=12.0):
        '''
        file_path: path to the dataset
        save_dir: directory to save the processed data
        num_entry_per_file: number of entries in a single file. -1 to save all data into one file 
                            (In-memory dataset)
        '''
        super().__init__()
        if save_dir is None:
            if not os.path.isdir(file_path):
                save_dir = os.path.split(file_path)[0]
            else:
                save_dir = file_path
            prefix = os.path.split(file_path)[1]
            if '.' in prefix:
                prefix = prefix.split('.')[0]
            save_dir = os.path.join(save_dir, f'{prefix}_processed')
        
        self.use_esm = use_esm
        if use_esm:
            save_dir += '_esm'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metainfo_file = os.path.join(save_dir, '_metainfo')
        self.data: List[AAComplex] = []  # list of ABComplex

        # try loading preprocessed files
        need_process = False
        try:
            with open(metainfo_file, 'r') as fin:
                metainfo = json.load(fin)
                self.num_entry = metainfo['num_entry']
                self.file_names = [os.path.join(save_dir, metainfo['file_names'][i].split('/')[-1]) for i in range(len( metainfo['file_names']))]
                self.file_num_entries = metainfo['file_num_entries']
        except Exception as e:
            print_log(f'Faild to load file {metainfo_file}, error: {e}', level='WARN')
            need_process = True

        if need_process:
            # preprocess
            self.file_names, self.file_num_entries = [], []
            self.preprocess(file_path, save_dir, num_entry_per_file, cdr_type=cdr_type)
            self.num_entry = sum(self.file_num_entries)

            metainfo = {
                'num_entry': self.num_entry,
                'file_names': self.file_names,
                'file_num_entries': self.file_num_entries
            }
            with open(metainfo_file, 'w') as fout:
                json.dump(metainfo, fout)

        self.random = random
        self.cur_file_idx, self.cur_idx_range = 0, (0, self.file_num_entries[0])  # left close, right open
        self._load_part()

        # user defined variables
        self.idx_mapping = [i for i in range(self.num_entry)]
        self.mode = '111'  # H/L/Antigen, 1 for include, 0 for exclude
        self.ctx_cutoff = ctx_cutoff
        self.interface_cutoff = interface_cutoff

    def _save_part(self, save_dir, num_entry):
        file_name = os.path.join(save_dir, f'part_{len(self.file_names)}.pkl')
        print_log(f'Saving {file_name} ...')
        file_name = os.path.abspath(file_name)
        if num_entry == -1:
            end = len(self.data)
        else:
            end = min(num_entry, len(self.data))
        with open(file_name, 'wb') as fout:
            pickle.dump(self.data[:end], fout)
        self.file_names.append(file_name)
        self.file_num_entries.append(end)
        self.data = self.data[end:]

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        print_log(f'Loading preprocessed file {f}, {self.cur_file_idx + 1}/{len(self.file_names)}')
        with open(f, 'rb') as fin:
            del self.data
            self.data = pickle.load(fin)
        self.access_idx = [i for i in range(len(self.data))]
        if self.random:
            np.random.shuffle(self.access_idx)

    def _check_load_part(self, idx):
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        idx = self.access_idx[idx - self.cur_idx_range[0]]
        return idx
     
    ########### load data from file_path and add to self.data ##########
    def preprocess(self, file_path, save_dir, num_entry_per_file, cdr_type='3'):
        '''
        Load data from file_path and add processed data entries to self.data.
        Remember to call self._save_data(num_entry_per_file) to control the number
        of items in self.data (this function will save the first num_entry_per_file
        data and release them from self.data) e.g. call it when len(self.data) reaches
        num_entry_per_file.
        '''
        with open(file_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        for line in tqdm(lines):
            item = json.loads(line)
            try:
                protein = Protein.from_pdb(os.path.join(PDB_PATH, item['pdb_data_path'].split('/')[-1]))
            except AssertionError as e:
                print_log(e, level='ERROR')
                print_log(f'parse {item["pdb"]} pdb failed, skip', level='ERROR')
                continue

            pdb_id, peptides = item['pdb'], protein.peptides
            self.data.append(AAComplex(pdb_id, peptides, item['heavy_chain'], item['light_chain'], item['antigen_chains'], cdr_type=cdr_type, use_esm=self.use_esm))
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

    def __getitem__(self, idx):
        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)
        item, res = self.data[idx], {}
        # each item is an instance of ABComplex. res has following entries
        # X: [seq_len, 4, 3], coordinates of N, CA, C, O. Missing data are set to the average of two adjacent nodes
        # S: [seq_len], indices of each residue
        # L: string of cdr labels, 0 for non-cdr residues, 1 for cdr1, 2 for cdr2, 3 for cdr3 

        hc, lc = item.get_heavy_chain(), item.get_light_chain()
        antigen_chains = item.get_antigen_chains(interface_only=True, cdr=None)

        # prepare input
        chain_lists, begins = [], []
        if self.mode[0] == '1': # the cdrh3 start pos in a batch should be near (for RefineGNN)
            chain_lists.append([hc])
            begins.append(VOCAB.BOH) # '+'
        if self.mode[1] == '1':
            chain_lists.append([lc])
            begins.append(VOCAB.BOL) # '-'
        if self.mode[2] == '1':
            chain_lists.append(antigen_chains)
            begins.append(VOCAB.BOA) # '&'
        
        X, S, = [], []
        X_l = []
        chain_start_ends = []  # tuples of [start, end)
        ESM = []
        # format input, box is begin of chain x
        corrupted_idx = []
        for chains, box in zip(chain_lists, begins):
            # judge if the chain has length
            skip = True
            for chain in chains:
                if len(chain):
                    skip = False
                    break
            if skip:
                continue
            X.append([(0, 0, 0) for _ in range(4)])  # begin symbol is global symbol, update coordination afterwards
            S.append(VOCAB.symbol_to_idx(box))
            ESM.append(np.zeros([1, 1280], dtype=np.float32))

            start = len(X)
            for chain in chains:
                if hasattr(chain, 'esm'):
                    ESM.append(chain.esm)
                for i in range(len(chain)):  # some chains do not participate
                    residue = chain.get_residue(i)
                    coord = residue.get_coord_map()
                    x = []
                    for atom in ['N', 'CA', 'C', 'O']:
                        if atom in coord:
                            x.append(coord[atom])
                        else:
                            coord[atom] = (0, 0, 0)
                            x.append((0, 0, 0))
                            corrupted_idx.append(len(X))
                            # print_log(f'Missing backbone atom coordination: {atom}', level='WARN')

                    X.append(np.array(x))
                    S.append(VOCAB.symbol_to_idx(residue.get_symbol()))
            X[start - 1] = np.mean(X[start:], axis=0)  # coordinate of global node
            chain_start_ends.append((start - 1, len(X)))

        # deal with corrupted coordinates
        for i in corrupted_idx:
            l, r = i - 1, i + 1
            if l > 0 and r < len(X):  # if at start / end, then leave it be
                X[i] = (X[l] + X[r]) / 2

        # set CDR pos for heavy chain
        offset = S.index(VOCAB.symbol_to_idx(VOCAB.BOH)) + 1
        L = ['0' for _ in X]

        for i in range(1, 4):
            begin, end = item.get_cdr_pos(f'H{i}')
            begin += offset
            end += offset
            for pos in range(begin, end + 1):
                L[pos] = str(i)

        res = {
            'X': torch.tensor(np.array(X), dtype=torch.float), # 3d coordination [n_node, 4, 3]
            'S': torch.tensor(S, dtype=torch.long),  # 1d sequence     [n_node]
            'L': ''.join(L),                          # cdr annotation, str of length n_node, 1 / 2 / 3 for cdr H1/H2/H3
            'ESM': torch.tensor(np.concatenate(ESM, axis=0))
        }
        
        return res

    def __len__(self):
        return self.num_entry


    @classmethod
    def collate_fn(cls, batch):
        Xs, Ss, Ls = [], [], []
        ESMs = []
        offsets = [0]
        for i, data in enumerate(batch):
            
            ## pass the no-head CDR
            if data['L'][1] == '1':
                continue

            Xs.append(data['X'])
            Ss.append(data['S'])
            Ls.append(data['L'])
            ESMs.append(data['ESM'])
            offsets.append(offsets[-1] + len(Ss[-1]))

        return {
            'X': torch.cat(Xs, dim=0),  # [n_all_node, 4, 3]
            'S': torch.cat(Ss, dim=0),  # [n_all_node]
            'ESMs': torch.cat(ESMs, dim=0),
            'L': Ls,
            'offsets': torch.tensor(offsets, dtype=torch.long)
        }
    

class ITAWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, n_samples, _cmp=lambda score1, score2: score1 - score2):
        super().__init__()
        self.dataset = deepcopy(dataset)
        self.dataset._check_load_part = lambda idx: idx
        self.candidates = [[(self.dataset.data[i], 0)] for i in self.dataset.idx_mapping]
        self.n_samples = n_samples
        self.cmp = _cmp

    def _cmp_wrapper(self, a, b):  # tuple of (cplx, score)
        return self.cmp(a[1], b[1])

    def update_candidates(self, i, candidates): # tuple of (cplx, score)
        all_cand = candidates + self.candidates[i]
        all_cand.sort(key=functools.cmp_to_key(self._cmp_wrapper))
        self.candidates[i] = all_cand[:self.n_samples]

    def finish_update(self):  # update all candidates to dataset
        data, idx_mapping = [], []
        for candidates in self.candidates:
            for cand, score in candidates:
                idx_mapping.append(len(data))
                data.append(cand)
        self.dataset.data = data
        self.dataset.idx_mapping = idx_mapping

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        return self.dataset.collate_fn(batch)



class ITAWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, n_samples, _cmp=lambda score1, score2: score1 - score2):
        super().__init__()
        self.dataset = deepcopy(dataset)
        self.dataset._check_load_part = lambda idx: idx
        self.candidates = [[(self.dataset.data[i], 0)] for i in self.dataset.idx_mapping]
        self.n_samples = n_samples
        self.cmp = _cmp

    def _cmp_wrapper(self, a, b):  # tuple of (cplx, score)
        return self.cmp(a[1], b[1])

    def update_candidates(self, i, candidates): # tuple of (cplx, score)
        all_cand = candidates + self.candidates[i]
        all_cand.sort(key=functools.cmp_to_key(self._cmp_wrapper))
        self.candidates[i] = all_cand[:self.n_samples]

    def finish_update(self):  # update all candidates to dataset
        data, idx_mapping = [], []
        for candidates in self.candidates:
            for cand, score in candidates:
                idx_mapping.append(len(data))
                data.append(cand)
        self.dataset.data = data
        self.dataset.idx_mapping = idx_mapping

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        return self.dataset.collate_fn(batch)