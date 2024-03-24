import os
import csv
import nni
import json
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from trainers import Trainer
from models import GeoRefiner
from dataset import EquiAACDataset, AAComplex

from evaluation import average_test, average_test_struct
from utils import set_seed, check_dir, write_result_to_file

key_list_kfold = ['PPL_mean', 'RMSD_mean', 'TMscore_mean', 'AAR_mean', 'PPL_std', 'RMSD_std', 'TMscore_std', 'AAR_std', 'PPL', 'RMSD', 'TMscore', 'AAR']
key_list = ['PPL', 'RMSD', 'TMscore', 'AAR']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Experiment:
    def __init__(self, args):
        self.args = args


    def generate(self, data_dir, save_dir, only_struct=False):

        model = torch.load(os.path.join(save_dir, 'checkpoint/best.ckpt')).to(device)

        test_set = EquiAACDataset(os.path.join(data_dir, 'test.json'), self.args.cdr_type, use_esm=self.args.use_esm)
        test_set.mode = self.args.mode
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, num_workers=4, shuffle=False, collate_fn=EquiAACDataset.collate_fn)
        
        model.eval()
        if only_struct:
            report_res = average_test_struct(self.args, model, test_set, test_loader, save_dir, device, run=self.args.run)
        else:
            report_res = average_test(self.args, model, test_set, test_loader, save_dir, device, run=self.args.run)
        
        if self.args.output_pdb == True:
            out_dir = os.path.join(save_dir, 'results', 'original')
            check_dir(out_dir)
            for cplx in tqdm(test_set.data):
                pdb_path = os.path.join(out_dir, cplx.get_id() + '.pdb')
                cplx.to_pdb(pdb_path)

        return report_res
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')

    parser.add_argument('--cdr_type', type=str, default='3', help='type of cdr')
    parser.add_argument('--mode', type=str, default='111', help='H/L/Antigen, 1 for include, 0 for exclude')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--data_root', type=str, default='./all_data', help='data root')
    parser.add_argument('--save_root', type=str, default='./results', help='save root')
    parser.add_argument('--eval_dir', type=str, default='H3_refine', help='save root for pretrained model')
    parser.add_argument('--use_esm', type=bool, default=False, help='Use esm to encode sequence')
    parser.add_argument('--run', type=int, default=100, help='run times')
    parser.add_argument('--only_struct', type=bool, default=False, help='if test only structure')

    parser.add_argument('--seed', type=int, default=2022, help='Seed to use in training')
    parser.add_argument('--output_pdb', type=bool, default=True, help='Whether to use save pdb files')


    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())
    args = argparse.Namespace(**param)
    timestamp = time.strftime("%Y-%m-%d %H-%M-%S") + f"-%3d" % ((time.time() - int(time.time())) * 1000)
    args.timestamp = timestamp

    set_seed(args.seed)
    exp = Experiment(args)
    
    save_dir = os.path.join(args.save_root + '/cdrh{}'.format(args.cdr_type), args.eval_dir)

    result = exp.generate(os.path.join(args.data_root, 'RAbD_H{}'.format(args.cdr_type)), 
                                save_dir,
                                only_struct=args.only_struct)
    print(result)