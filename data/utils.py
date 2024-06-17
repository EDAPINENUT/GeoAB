import os
import sys
import torch
import shutil
import random
import datetime
import numpy as np


LEVELS = ['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR']
LEVELS_MAP = None


def init_map():
    global LEVELS_MAP, LEVELS
    LEVELS_MAP = {}
    for idx, level in enumerate(LEVELS):
        LEVELS_MAP[level] = idx


def get_prio(level):
    global LEVELS_MAP
    if LEVELS_MAP is None:
        init_map()
    return LEVELS_MAP[level.upper()]


def print_log(s, level='INFO', end='\n', no_prefix=False):
    pth_prio = get_prio(os.getenv('LOG', 'INFO'))
    prio = get_prio(level)
    if prio >= pth_prio:
        if not no_prefix:
            now = datetime.datetime.now()
            prefix = now.strftime("%Y-%m-%d %H:%M:%S") + f'::{level.upper()}::'
            print(prefix, end='')
        print(s, end=end)
        sys.stdout.flush()

        
def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False


def check_dir(path, overwrite=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


def save_code(save_dir):
    save_code_dir = os.path.join(save_dir, 'codes/')
    check_dir(save_code_dir)

    for file in os.listdir("code"):
        if '.py' in file:
            shutil.copyfile('code/' + file, save_code_dir + file)


def valid_check(seq):
    charge_valid, motif_valid, seq_valid = True, True, True
    
    # charge
    charge = 0
    for res in seq:
        if res == 'R' or res == 'K':
            charge += 1
        elif res == 'H':
            charge += 0.1
        elif res == 'D' or res == 'E':
            charge -= 1
    if charge < -2.0 or charge > 2.0:
        charge_valid = False

    # motif
    for i in range(len(seq) - 2):
        motif = seq[i:i+3]
        if motif[0] == 'N' and (motif[-1] == 'S' or motif[-1] == 'T'):
            motif_valid = False
            break
        
    # seq
    longest, previous, cnt = 0, None, 0
    for res in seq:
        if res == previous:
            cnt += 1
            longest = max(longest, cnt)
        else:
            cnt = 1
        previous = res
    if longest > 5:
        seq_valid = False
    
    return motif_valid and charge_valid and seq_valid