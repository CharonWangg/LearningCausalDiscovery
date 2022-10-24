from causallearn.search.ScoreBased.GES import ges
from causallearn.search.Granger.Granger import Granger
from causallearn.search.FCMBased import lingam
from collections import defaultdict
import tsaug
import pickle
import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import scipy.io as scio
from entropy_estimators import mi
import sys
from copy import deepcopy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shallowmind.api.infer import prepare_inference

import warnings
warnings.filterwarnings("ignore")


def parse_examples(sim, percentage, add_noise=False):
    if add_noise:
        noiser = tsaug.AddNoise(scale=0.5, normalize=True, seed=42).augment
    data = scio.loadmat(sim)
    n_subjects, n_node, duration = int(data['Nsubjects']), int(data['Nnodes']), int(data['Ntimepoints'])
    slice = [int(percentage[0] * n_subjects), int(percentage[1] * n_subjects)]
    data['ts'] = data['ts'][duration * slice[0]:duration * slice[1]]
    data['net'] = data['net'][slice[0]:slice[1]]
    examples = []
    for i, start in enumerate(range(0, data['ts'].shape[0], duration)):
        if add_noise:
            examples.append({
                'sample_id': i,
                'seqs': noiser(data['ts'][start:start + duration]),
                'label': data['net'][i]
            })
        else:
            examples.append({
                'sample_id': i,
                'seqs': data['ts'][start:start + duration],
                'label': data['net'][i]
            })
    return examples


def eval_by_ges(examples):
    res = []
    for example in tqdm(examples, desc='GES'):
        Record = ges(example['seqs'], score_func='local_score_BIC')
        g = Record['G'].graph
        adj_g = np.array(
            [[1 if g[j, i] == 1 and g[i, j] == -1 else 0 for j in range(g.shape[1])] for i in range(g.shape[0])])
        # do not need to predict the diag elements
        mask = np.eye(example['label'].shape[-1], dtype=bool)
        res.append({'pred': adj_g[~mask],
                    'label': example['label'][~mask]})
    return res


def eval_by_gc(examples, maxlag=5):
    G = Granger(maxlag=maxlag)
    res = []
    for example in tqdm(examples, desc='GC'):
        N = example['label'].shape[-1]
        # select the last maxlag matrix
        coeff = G.granger_lasso(example['seqs'])
        # do not need to predict the diag elements
        mask = np.eye(N, dtype=bool)
        # calculate the largest absolute weight value of each edge across lags
        coeff = np.array([[np.max(np.abs(coeff[i, j::N])) for j in range(N)] for i in range(N)]).T
        res.append({'pred': np.where(coeff!=0, 1, 0)[~mask],
                    'label': example['label'][~mask]})

    return res


def eval_by_lingam(examples, random_state=42):
    res = []
    for example in tqdm(examples, desc='LiNGAM'):
        model = lingam.ICALiNGAM(random_state=random_state)
        model.fit(example['seqs'])
        # do not need to predict the diag elements
        mask = np.eye(example['label'].shape[-1], dtype=bool)

        res.append({'pred': np.where(model.adjacency_matrix_!=0, 1, 0).T[~mask],
                    'label': example['label'][~mask]})
    return res


def eval_by_mi(examples):
    res = []
    for example in tqdm(examples, desc='MI'):
        temp_mat = []
        temp_label = []
        for i in range(example['label'].shape[-1]):
            for j in range(example['label'].shape[-1]):
                # do not need to predict the diag elements
                if i == j: continue
                temp_mat.append(mi(example['seqs'][:, i], example['seqs'][:, j]))
                temp_label.append(example['label'][i, j])
        res.append({'pred': temp_mat, 'label': temp_label})

    return res


def eval_by_sldisco(examples):
    res = []
    # SLdisco inference
    # path of the SLdisco model's checkpoint and the config file
    if examples[0]['label'].shape[-1] == 5:
        cfg = 'work_dir/sldisco_node_5_n_samples_10000/sldisco_node_5_n_samples_10000.py'
        ckpt = 'work_dir/sldisco_node_5_n_samples_10000/ckpts/exp_name=sldisco_node_5_n_samples_10000-cfg=sldisco_node_5_n_samples_10000-bs=1024-seed=42-val_loss_epoch=0.2563.ckpt'
    elif examples[0]['label'].shape[-1] == 10:
        cfg = 'work_dir/sldisco_node_10_n_samples_10000/sldisco_node_10_n_samples_10000.py'
        ckpt = 'work_dir/sldisco_node_10_n_samples_10000/ckpts/exp_name=sldisco_node_10_n_samples_10000-cfg=sldisco_node_10_n_samples_10000-bs=1024-seed=42-val_loss_epoch=0.4550.ckpt'
    elif examples[0]['label'].shape[-1] == 15:
        cfg = 'work_dir/sldisco_node_15_n_samples_10000/sldisco_node_15_n_samples_10000.py'
        ckpt = 'work_dir/sldisco_node_15_n_samples_10000/ckpts/exp_name=sldisco_node_15_n_samples_10000-cfg=sldisco_node_15_n_samples_10000-bs=1024-seed=42-val_loss_epoch=0.4709.ckpt'
    elif examples[0]['label'].shape[-1] == 50:
        cfg = 'work_dir/sldisco_node_50_n_samples_10000/sldisco_node_50_n_samples_10000.py'
        ckpt = 'work_dir/sldisco_node_50_n_samples_10000/ckpts/exp_name=sldisco_node_50_n_samples_10000-cfg=sldisco_node_50_n_samples_10000-bs=1024-seed=42-val_loss_epoch=0.4883.ckpt'

    di, mi = prepare_inference(cfg, ckpt)
    mi = mi.cuda()
    mi.eval()
    with torch.no_grad():
        for example in tqdm(examples, desc='SLDisco'):
            ts = example['seqs']
            windows = np.corrcoef(ts, rowvar=False).reshape(1, 1, example['label'].shape[-1], example['label'].shape[-1])
            x = torch.tensor(windows).float().cuda()
            pred = mi(x).sigmoid()
            # do not need to predict the diag elements
            mask = torch.eye(example['label'].shape[-1], dtype=torch.bool)
            pred = pred.reshape(example['label'].shape[-1], example['label'].shape[-1])
            res.append({'pred': pred.cpu()[~mask], 'label': example['label'][~mask]})
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='.cache/sim_data/')
    # noise-free or add noise
    parser.add_argument('--condition', type=str, choices=['default', 'noise'], default='default')
    args = parser.parse_args()
    root_dir = args.root_dir

    pred_sldisco, pred_ges, pred_gc, pred_mi, pred_lingam = \
        defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    percentage = [0.6, 1.0]
    netsim_dir = os.path.join(os.path.dirname(os.path.dirname(root_dir)), 'netsim')
    # all simulations have fixed sample length 200
    sims = [f'{netsim_dir}/sim{i}.mat' for i in [1, 2, 3, 4, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24]]

    for sim in tqdm(sims, desc='Sim'):
        examples = parse_examples(sim, percentage, add_noise=(args.condition=='noise'))

        if 'sim4' in sim:
            # sldisco
            pred_sldisco[sim] = eval_by_sldisco(examples)
            # # ges
            pred_ges[sim] = []
            # # granger causality
            pred_gc[sim] = []
            # # mutual info
            pred_mi[sim] = eval_by_mi(examples)
            # lingam
            pred_lingam[sim] = eval_by_lingam(examples)
            continue
        # sldisco
        pred_sldisco[sim] = eval_by_sldisco(examples)
        # # ges
        pred_ges[sim] = eval_by_ges(examples)
        # # granger causality
        pred_gc[sim] = eval_by_gc(examples)
        # # mutual info
        pred_mi[sim] = eval_by_mi(examples)
        # lingam
        pred_lingam[sim] = eval_by_lingam(examples)

    # save
    if not os.path.exists(os.path.join(root_dir, 'netsim_auc_result')):
        os.makedirs(os.path.join(root_dir, 'netsim_auc_result'))
    if not os.path.exists(os.path.join(root_dir, 'netsim_auc_result', f'{args.condition}_result.pkl')):
        pickle.dump({'sldisco': pred_sldisco, 'ges': pred_ges, 'gc': pred_gc, 'mi': pred_mi, 'lingam': pred_lingam},
                    open(os.path.join(root_dir, 'netsim_auc_result', f'{args.condition}_result.pkl'), 'wb'))
    else:
        print('File already exist')