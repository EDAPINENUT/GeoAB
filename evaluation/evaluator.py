import os
import json
import torch
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from functools import partial
from tqdm.contrib.concurrent import process_map
from data.pdb_utils import VOCAB, Residue, Peptide, Protein
from copy import copy
from utils import print_log
from data.pdb_utils import AAComplex
from .tm_score import tm_score
from .rmsd import kabsch, compute_rmsd
import time
import re

PROJ_DIR = os.path.split(__file__)[0]
CACHE_DIR = os.path.join(PROJ_DIR, '__cache__')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def exec_bin(mod_pdb, ref_pdb, log, backbone_only):
    options = '-x'
    if backbone_only:
        options += ' -c'
    cmd = f'lddt {options} {mod_pdb} {ref_pdb} > {log} 2>&1'
    return os.system(cmd)

def merge_to_one_chain(protein: Protein):
    residues = []
    chain_order = sorted(protein.get_chain_names())
    for chain_name in chain_order:
        chain = protein.get_chain(chain_name)
        for _, residue in enumerate(chain.residues):
            residue.id = (len(residues), ' ')
            residues.append(residue)
    return Protein(protein.get_id(), {'A': Peptide('A', residues)})

def lddt(mod_protein: Protein, ref_protein: Protein, backbone_only=False):
    # concatenate all chains to one chain
    mod_protein = merge_to_one_chain(mod_protein)
    ref_protein = merge_to_one_chain(ref_protein)

    mod_sign, ref_sign = id(mod_protein), id(ref_protein)
    mod_pdb = os.path.join(CACHE_DIR, f'lddt_{mod_sign}_mod_{time.time()}.pdb')
    ref_pdb = os.path.join(CACHE_DIR, f'lddt_{ref_sign}_ref_{time.time()}.pdb')
    log = os.path.join(CACHE_DIR, f'lddt_log_{mod_sign}_{ref_sign}.txt')
    
    mod_protein.to_pdb(mod_pdb)
    ref_protein.to_pdb(ref_pdb)

    res_code = exec_bin(mod_pdb, ref_pdb, log, backbone_only=True)
    if res_code != 0:
        raise ValueError(f'lddt execution failed')
    with open(log, 'r') as fin:
        text = fin.read()
    res = re.search(r'Global LDDT score: ([0-1]\.?[0-9]*)', text)
    score = float(res.group(1))
    os.remove(mod_pdb)
    os.remove(ref_pdb)
    os.remove(log)
    return score, text

def set_cdr(cplx, seq, x, cdr='H3'):

    cdr = cdr.upper()
    cplx: AAComplex = deepcopy(cplx)
    chains = cplx.peptides
    cdr_chain_key = cplx.heavy_chain if 'H' in cdr else cplx.light_chain
    refined_chain = chains[cdr_chain_key]

    start, end = cplx.get_cdr_pos(cdr)
    start_pos, end_pos = refined_chain.get_ca_pos(start), refined_chain.get_ca_pos(end)
    start_trans, end_trans = x[0][1] - start_pos, x[-1][1] - end_pos

    # left to start of cdr
    for i in range(0, start):
        refined_chain.set_residue_translation(i, start_trans)

    # end of cdr to right
    for i in range(end + 1, len(refined_chain)):
        refined_chain.set_residue_translation(i, end_trans)

    # cdr 
    for i, residue_x, symbol in zip(range(start, end + 1), x, seq):
        center = residue_x[4] if len(residue_x) > 4 else None
        refined_chain.set_residue(i, symbol,
            {
                'N': residue_x[0],
                'CA': residue_x[1],
                'C': residue_x[2],
                'O': residue_x[3]
            }, center, gen_side_chain=False
        )

    new_cplx = AAComplex(cplx.pdb_id, chains, cplx.heavy_chain, cplx.light_chain,
                         cplx.antigen_chains, numbering=None, cdr_type=cdr[-1], cdr_pos=cplx.cdr_pos,
                         skip_cal_interface=True)
    
    return new_cplx


def eval_one(tup, out_dir, cdr='H3', output_pdb=False):
    cplx, seq, x, true_x, aligned = tup
    ref_cplx = copy(cplx)

    summary = {
        'pdb': cplx.get_id(),
        'heavy_chain': cplx.heavy_chain,
        'light_chain': cplx.light_chain,
        'antigen_chains': cplx.antigen_chains
    }

    # kabsch
    if aligned:
        ca_aligned = x[:, 1, :]
    else:
        ca_aligned, rotation, t = kabsch(x[:, 1, :], true_x[:, 1, :])
        x = np.dot(x - np.mean(x, axis=0), rotation) + t
    summary['RMSD'] = compute_rmsd(ca_aligned, true_x[:, 1, :], aligned=True)
    summary['RMSD_Full'] = compute_rmsd(x, true_x, aligned=True)
    summary['RMSD_Align'] = compute_rmsd(ca_aligned, true_x[:, 1, :], aligned=False)
    # set cdr
    new_cplx = set_cdr(cplx, seq, x, cdr)
    score, _ = lddt(new_cplx, ref_cplx)
    summary['LDDT'] = score

    if output_pdb:
        pdb_path = os.path.join(out_dir, cplx.get_id() + '.pdb')
        new_cplx.to_pdb(pdb_path)
    summary['TMscore'] = tm_score(cplx.get_heavy_chain(), new_cplx.get_heavy_chain())

    # AAR
    origin_seq = cplx.get_cdr(cdr).get_seq()
    hit = 0
    for a, b in zip(origin_seq, seq):
        if a == b:
            hit += 1
    aar = hit * 1.0 / len(origin_seq)
    summary['AAR'] = aar

    return summary



def to_cplx(ori_cplx, ab_x, ab_s):
    heavy_chain, light_chain = [], []
    chain = None
    for residue, residue_x in zip(ab_s, ab_x):
        residue = VOCAB.idx_to_symbol(residue)
        if residue == VOCAB.BOA:
            continue
        elif residue == VOCAB.BOH:
            chain = heavy_chain
            continue
        elif residue == VOCAB.BOL:
            chain = light_chain
            continue
        if chain is None:  # still in antigen region
            continue
        coord, atoms = {}, VOCAB.backbone_atoms 

        for atom, x in zip(atoms, residue_x):
            coord[atom] = x
        chain.append(Residue(
            residue, coord, _id=(len(chain), ' ')
        ))
    heavy_chain = Peptide(ori_cplx.heavy_chain, heavy_chain)
    light_chain = Peptide(ori_cplx.light_chain, light_chain)
    for res, ori_res in zip(heavy_chain, ori_cplx.get_heavy_chain()):
        res.id = ori_res.id
    for res, ori_res in zip(light_chain, ori_cplx.get_light_chain()):
        res.id = ori_res.id

    peptides = {
        ori_cplx.heavy_chain: heavy_chain,
        ori_cplx.light_chain: light_chain,
    }
    antibody = Protein(ori_cplx.pdb_id, peptides)
    cplx = AAComplex(
        ori_cplx.pdb_id, antibody, ori_cplx.heavy_chain,
        ori_cplx.light_chain, []
    )
    cplx.cdr_pos = ori_cplx.cdr_pos
    return cplx


def average_test_struct(args, model, test_set, test_loader, save_dir, device, run=1):
    heads, eval_res = ['RMSD', 'TMscore', 'AAR'], []
    summaries_meta = []
    for _round in range(run):
        results = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                true_S, xs, true_xs, aligned = model.infer(batch, device)
                results.extend([(true_S[i], xs[i], true_xs[i], aligned) for i in range(len(xs))])

        assert len(test_set) == len(results)
        out_dir = os.path.join(save_dir, 'results', "Run " + str(_round+1))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        inputs = [(cplx, ) + item for cplx, item in zip(test_set.data, results)]
        summaries = process_map(partial(eval_one, out_dir=out_dir, cdr='H'+model.cdr_type, output_pdb=args.output_pdb), inputs, max_workers=4, chunksize=10)
        summaries_meta.extend(summaries)

        if args.output_pdb == True:
            summary_fout = open(os.path.join(out_dir, 'summary.json'), 'w')
            for i, summary in enumerate(summaries):
                summary_fout.write(json.dumps(summary) + '\n')
            summary_fout.close()

        rmsds = [summary['RMSD'] for summary in summaries_meta]
        tm_scores = [summary['TMscore'] for summary in summaries_meta]
        aars = [summary['AAR'] for summary in summaries_meta]
        rmsd, tm, aar = np.mean(rmsds), np.mean(tm_scores), np.mean(aars)
        
        print_log(f'rmsd: {rmsd}, TM score: {tm}, AAR: {aar}')
        eval_res.append([rmsd, tm, aar])

    eval_res = np.array(eval_res)
    means = np.mean(eval_res, axis=0)

    return {heads[i]: means[i] for i, h in enumerate(heads)}


def average_test(args, model, test_set, test_loader, save_dir, device, run=1):
    heads = ['PPL', 'RMSD', 'TMscore', 'AAR', 'RMSD_Full', 'RMSD_Align', 'LDDT']
    summaries_meta = []
    for _round in range(run):
        results, ppl = [], []
        meta_for_save = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                ppls, seqs, xs, true_xs, aligned, meta_data = model.infer(batch, device)
                ppl.extend(ppls)
                results.extend([(seqs[i], xs[i], true_xs[i], aligned) for i in range(len(seqs))])
                meta_for_save.append(meta_data)

        out_dir = os.path.join(save_dir, 'results', "prediction")
        if not os.path.exists(out_dir) and args.output_pdb:
            os.makedirs(out_dir)

        inputs = [(cplx, ) + item for cplx, item in zip(test_set.data, results)]

        summaries = process_map(partial(eval_one, out_dir=out_dir, cdr='H'+model.cdr_type, output_pdb=args.output_pdb), inputs, max_workers=4, chunksize=10)
        summaries_meta.extend(summaries)
        if args.output_pdb == True:
            meta_for_save = merge_meta(meta_for_save)
            torch.save(meta_for_save, os.path.join(out_dir, 'meta.pt'))

            summary_fout = open(os.path.join(out_dir, 'summary.json'), 'w')
            for i, summary in enumerate(summaries):
                summary['PPL'] = ppl[i]
                summary_fout.write(json.dumps(summary) + '\n')
            summary_fout.close()

    rmsds = [summary['RMSD'] for summary in summaries_meta]
    tm_scores = [summary['TMscore'] for summary in summaries_meta]
    aars = [summary['AAR'] for summary in summaries_meta]
    rmsds_full = [summary['RMSD_Full'] for summary in summaries_meta]
    rmsds_align = [summary['RMSD_Align'] for summary in summaries_meta]
    lddts = [summary['LDDT'] for summary in summaries_meta]
    if args.output_pdb == True:
        out_dict = {'rmsds': np.array(rmsds).reshape(run, -1), 
        'tm_scores': np.array(tm_scores).reshape(run, -1), 
        'aars': np.array(aars).reshape(run, -1), 
        'rmsds_full': np.array(rmsds_full).reshape(run, -1), 
        'rmsds_align': np.array(rmsds_align).reshape(run, -1), 
        'lddts': np.array(lddts).reshape(run, -1)}
        np.save(os.path.join(out_dir, 'out_dict.npy'), out_dict)

    ppl_mean, rmsd, tm, aar, rmsd_full, rmsd_align, lddt_score = np.mean(ppl), np.mean(rmsds), np.mean(tm_scores), np.mean(aars), np.mean(rmsds_full),np.mean(rmsds_align), np.mean(lddts)
    
    print_log(f'ppl: {ppl_mean}, rmsd: {rmsd}, TM score: {tm}, AAR: {aar}, rmsd align: {rmsd_align}, lddt_score: {lddt_score}')
    means = [ppl_mean, rmsd, tm, aar, rmsd_full, rmsd_align, lddt_score]

    return {heads[i]: means[i] for i, _ in enumerate(heads)}


def merge_meta(meta_data):
    meta = {}
    for m in meta_data:
        for key in m:
            if key not in meta:
                meta[key] = []
            meta[key].append(m[key])
    return meta