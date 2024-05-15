import os
import csv
from typing import Any
import json
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from trainers import Trainer
from models import GeoRefiner
from dataset import EquiAACDataset

from evaluation import average_test, set_cdr
from utils import set_seed, check_dir, print_log, save_code, valid_check, write_result_to_file, EMAWeight
from functools import partial
key_list_kfold = ['PPL_mean', 'RMSD_mean', 'TMscore_mean', 'AAR_mean', 'PPL_std', 'RMSD_std', 'TMscore_std', 'AAR_std', 'PPL', 'RMSD', 'TMscore', 'AAR']
key_list = ['PPL', 'RMSD', 'TMscore', 'AAR']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def constant_beta(iter_num, beta):
    return beta

class Experiment:
    def __init__(self, args):
        self.args = args
        self.use_esm = args.use_esm

    def _get_data_and_model(self, train_path, valid_path, cdr_type, initializer_path=None):

        train_set = EquiAACDataset(train_path, cdr_type, use_esm=self.use_esm)
        train_set.mode = self.args.mode
        valid_set = EquiAACDataset(valid_path, cdr_type, use_esm=self.use_esm)
        valid_set.mode = self.args.mode

        if initializer_path is not None:
            print('Loading pretrained initializer from {}'.format(os.path.join(initializer_path, 'checkpoint/best.ckpt')))
            initializer = torch.load(os.path.join(initializer_path, 'checkpoint/best.ckpt')).to(device)
        else:
            initializer = None
        if self.args.dybeta:
            beta = EMAWeight(decay=0.999, exp_decay=0.9999, beta_max=self.args.beta, beta_min=self.args.beta/10)
        else:
            beta = partial(constant_beta, beta=self.args.beta)

        n_channel = valid_set[0]['X'].shape[1]
        model = GeoRefiner(
                self.args.embed_size, 
                self.args.hidden_size, 
                n_channel, 
                n_layers=self.args.n_layers, 
                dropout=args.dropout,
                cdr_type=self.args.cdr_type, 
                alpha=self.args.alpha, 
                n_iter=self.args.n_iter,
                node_feats_mode=self.args.node_feats_mode, 
                edge_feats_mode=self.args.edge_feats_mode,
                interface_only=self.args.interface_only,
                local_update=self.args.use_local_update,
                n_layers_update=self.args.n_layers_update,
                beta = beta,
                use_esm=self.use_esm,
                initializer=initializer if initializer_path is not None else None
        )

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, num_workers=4, shuffle=True, collate_fn=EquiAACDataset.collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=self.args.batch_size, num_workers=4, shuffle=False, collate_fn=EquiAACDataset.collate_fn)
        
        return train_loader, valid_loader, model

    def generate(self, data_dir, save_dir):

        model = torch.load(os.path.join(save_dir, 'checkpoint/best.ckpt')).to(device)

        test_set = EquiAACDataset(os.path.join(data_dir, 'test.json'), cdr_type=self.args.cdr_type, use_esm=self.use_esm)
        test_set.mode = self.args.mode
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, num_workers=4, shuffle=False, collate_fn=EquiAACDataset.collate_fn)
        
        model.eval()
        report_res = average_test(self.args, model, test_set, test_loader, save_dir, device)

        if self.args.output_pdb == True:
            out_dir = os.path.join(save_dir, 'results', 'original')
            check_dir(out_dir)
            for cplx in tqdm(test_set.data):
                pdb_path = os.path.join(out_dir, cplx.get_id() + '.pdb')
                cplx.to_pdb(pdb_path)

        return report_res
    
    def train_eval(self, timestamp, eval_dir=None):
        print_log('CDR {}'.format(self.args.cdr_type))

        data_dir = os.path.join(self.args.data_root, 'RAbD_H{}'.format(self.args.cdr_type))
        save_dir = os.path.join(self.args.save_root + '/cdrh{}'.format(self.args.cdr_type), timestamp)
        check_dir(save_dir)
        # save_code(save_dir)
        
        train_loader, valid_loader, model = self._get_data_and_model(os.path.join(data_dir, 'train.json'), 
                                                            os.path.join(data_dir, 'valid.json'),
                                                            args.cdr_type,
                                                            args.initializer_path)

        trainer = Trainer(model, train_loader, valid_loader, save_dir, args)
        if not eval_dir:
            trainer.train()
        else:
            save_dir = os.path.join(self.args.save_root + '/cdrh{}'.format(self.args.cdr_type), eval_dir)
        report_res = self.generate(data_dir, save_dir)
        for key in report_res.keys():
            print_log('CDR {}| '.format(self.args.cdr_type) + f'{key}: {report_res[key]}')
            trainer.log_file.write('CDR {}| '.format(self.args.cdr_type) + f'{key}: {report_res[key]}\n')
            trainer.log_file.flush()

        return report_res

    def k_fold_train_eval(self, timestamp):

        res_dict = {'PPL': [], 'RMSD': [], 'TMscore': [], 'AAR': []}

        for k in range(10):
            print_log('CDR {}, Fold {}'.format(self.args.cdr_type, k))

            if self.args.split == -1:
                data_dir = os.path.join('./summaries/cdrh{}'.format(self.args.cdr_type), 'fold_{}'.format(k))
            else:
                data_dir = os.path.join('./summaries/data/spilt_{}/cdrh{}'.format(self.args.split, self.args.cdr_type), 'fold_{}'.format(k))
            save_dir = os.path.join('./results/cdrh{}'.format(self.args.cdr_type), 'fold_{}'.format(k), timestamp)
            check_dir(save_dir)
            save_code(save_dir)
            
            train_loader, valid_loader, model = self._get_data_and_model(os.path.join(data_dir, 'train.json'), 
                                                               os.path.join(data_dir, 'valid.json'),
                                                               args.initializer_path)

            trainer = Trainer(model, train_loader, valid_loader, save_dir, args)
            trainer.train()

            report_res = self.generate(data_dir, save_dir)
            for key in res_dict.keys():
                res_dict[key].append(report_res[key])
                print_log('CDR {}, Fold {} | '.format(self.args.cdr_type, k) + f'{key}: {report_res[key]}')
                trainer.log_file.write('CDR {}, Fold {} | '.format(self.args.cdr_type, k) + f'{key}: {report_res[key]}\n')
                trainer.log_file.flush()

        write_buffer = {}
        for key in res_dict.keys():
            vals = res_dict[key]
            val_mean, val_std = np.mean(vals), np.std(vals)
            write_buffer[key] = res_dict[key]
            write_buffer[key+'_mean'] = val_mean
            write_buffer[key+'_std'] = val_std
            print_log('CDR {} | '.format(self.args.cdr_type) + f'{key}: mean {val_mean}, std {val_std}')

        with open(os.path.join(save_dir, "eval_results.json"), "w") as f:
            json.dump(write_buffer, f, indent=2)

        return write_buffer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')

    parser.add_argument('--cdr_type', type=str, default='3', help='type of cdr')
    parser.add_argument('--mode', type=str, default='111', help='H/L/Antigen, 1 for include, 0 for exclude')
    parser.add_argument('--node_feats_mode', type=str, default='1111')
    parser.add_argument('--edge_feats_mode', type=str, default='1111')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--max_epoch', type=int, default=20, help='max training epoch')
    parser.add_argument('--data_root', type=str, default='./all_data', help='data root')
    parser.add_argument('--save_root', type=str, default='./results', help='save root')
    
    parser.add_argument('--embed_size', type=int, default=64, help='embed size of amino acids')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--n_layers', type=int, default=9, help='number of layers')
    parser.add_argument('--alpha', type=float, default=0.8, help='scale mse loss of coordinates')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--use_local_update', type=bool, default=True)
    parser.add_argument('--n_layers_update', type=int, default=3, help='number of layers')
    parser.add_argument('--beta', type=float, default=0.4, help='loss weight of local update geometry')
    parser.add_argument('--dybeta', type=bool, default=False, help=' if use dynamic loss weight of local update')

    parser.add_argument('--seed', type=int, default=2022, help='Seed to use in training')
    parser.add_argument('--early_stop', type=bool, default=True, help='Whether to use early stop')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--anneal_base', type=float, default=0.95, help='Exponential lr decay, 1 for not decay')
    parser.add_argument('--output_pdb', type=bool, default=False, help='Whether to use save pdb files')
    parser.add_argument('--interface_only', type=int, default=0, help='antigen interface_only')
    parser.add_argument('--split', type=int, default=-1, help='Which split used to train')
    parser.add_argument('--k_fold_eval', type=bool, default=False, help='Use k-fold training and evaluation')
    parser.add_argument('--use_esm', type=bool, default=False, help='Use esm to encode sequence')
    parser.add_argument('--tag', type=str, default='H3_refine', help='Use esm to encode sequence')

    parser.add_argument('--optimization', type=int, default=0, help='used for antibody optimization')
    parser.add_argument('--ita_epoch', type=int, default=1, help='number of epochs per iteration')
    parser.add_argument('--n_iter', type=int, default=1, help='Number of iterations to run')   
    parser.add_argument('--n_tries', type=int, default=50, help='Number of tries each iteration')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of samples each iteration')
    parser.add_argument('--update_freq', type=int, default=4, help='Model update frequency')
    parser.add_argument('--initializer_path', type=str, default=None, help='Path to pretrained model') # '/linhaitao/GeoAB/results/cdrh3/H3_generate/'

    args = parser.parse_args()
    param = args.__dict__
    args = argparse.Namespace(**param)
    timestamp = time.strftime("%Y-%m-%d %H-%M-%S") + f"-%3d" % ((time.time() - int(time.time())) * 1000)
    args.timestamp = timestamp

    set_seed(args.seed)
    exp = Experiment(args)
    result = exp.k_fold_train_eval(args.tag) if args.k_fold_eval else exp.train_eval(args.tag)
    print(result)
    
