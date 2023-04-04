import pandas as pd
from causallearn.search.Granger.Granger import Granger
from causallearn.search.FCMBased import lingam
from tigramite.independence_tests.parcorr import ParCorr
from causalnex.structure.dynotears import from_pandas_dynamic
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from external.Neural_GC.models import cmlp as _cmlp
from external.Neural_GC.models import clstm as _clstm
from external.SRU_for_GCI.utils.utilFuncs import loadTrainingData, loadTrueNetwork
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

import warnings

warnings.filterwarnings("ignore")


def parse_examples(data_paths):
    Xtrain = loadTrainingData(data_paths[0], 'cpu')
    n, l = Xtrain.shape
    Gref = loadTrueNetwork(data_paths[1], n).T  # entry (i,j) is the causal link i -> j

    # Load data
    duration = 21
    examples = []

    # parse it into 46 snippets of sequences (the length of each one is 21 )
    for start in range(0, l, duration):
        examples.append(
            {
                "seqs": Xtrain[:, start:start + duration],
                "label": Gref,
            }
        )
    return examples


def eval_by_gc(examples, maxlag=2):
    G = Granger(maxlag=maxlag)
    res = []
    for example in tqdm(examples, desc="GC"):
        N = example["label"].shape[-1]
        # select the last maxlag matrix
        coeff = G.granger_lasso(example["seqs"])
        # calculate the largest absolute weight value of each edge across lags
        coeff = np.array(
            [[np.max(np.abs(coeff[i, j::N])) for j in range(N)] for i in range(N)]
        ).T
        # ignore the diag predictions
        for i in range(N):
            coeff[i, i] = 0
        res.append(
            {
                "pred": coeff,
                "label": example["label"],
            }
        )

    return res


def eval_by_lingam(examples, max_lag=2, random_state=42):
    res = []
    for example in tqdm(examples, desc="LiNGAM"):
        N = example["label"].shape[-1]
        model = lingam.VARLiNGAM(lags=max_lag, random_state=random_state)
        model.fit(example["seqs"])
        adj = model.adjacency_matrices_  # shape: (max_lag, n_features, n_features)
        # take the max absolute weight value across lags
        lingam_score = np.max(np.abs(adj), axis=0).T
        # ignore the diag predictions
        for i in range(N):
            lingam_score[i, i] = 0
        res.append(
            {
                "pred": lingam_score,
                "label": example["label"],
            }
        )
    return res


def eval_by_mi(examples):
    res = []
    for example in tqdm(examples, desc="MI"):
        N = example["label"].shape[-1]
        score = np.zeros((N, N))
        for i in range(example["label"].shape[-1]):
            for j in range(example["label"].shape[-1]):
                score[i, j] = mi(example["seqs"][:, i], example["seqs"][:, j])
                # do not need to predict the diag elements
                if i == j:
                    continue
        # ignore the diag predictions
        for i in range(N):
            score[i, i] = 0

        res.append({"pred": score, "label": example["label"]})

    return res


def eval_by_pcmciplus(examples, max_lag=3):
    res = []
    for example in tqdm(examples, desc="PCMCI+"):
        X = example["seqs"]
        T, N = X.shape
        var_names = [f'X_{j}' for j in range(N)]
        df = pp.DataFrame(X, var_names=var_names)
        pcmci = PCMCI(dataframe=df,
                      cond_ind_test=ParCorr(significance='analytic'),
                      verbosity=0)
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=max_lag)
        # Initialize the summary graph as an N x N matrix with zeros
        summary_graph = np.zeros((N, N), dtype=int)
        # Iterate through the matrix G and fill in the summary graph
        for i in range(N):
            for j in range(N):
                for tau in range(max_lag + 1):
                    if results['graph'][i, j, tau] == '-->':
                        summary_graph[i, j] = 1
                        break

        # ignore the diag predictions
        for i in range(N):
            summary_graph[i, i] = 0
        res.append(
            {
                "pred": summary_graph,
                "label": example["label"],
            }
        )

    return res


def eval_by_dynotears(examples, max_lag=2):
    """
    modified from https://github.com/ckassaad/causal_discovery_for_time_series/blob/master/baselines/scripts_python/dynotears.py
    """
    res = []
    for example in tqdm(examples, desc="Dynotears"):
        X = example["seqs"]
        T, N = X.shape
        var_names = [f'X_{j}' for j in range(N)]

        df = pd.DataFrame(X, columns=var_names)
        sm = from_pandas_dynamic(df, lambda_w=0.5, lambda_a=0.5, max_iter=2000, p=max_lag)

        tname_to_name_dict = dict()
        count_lag = 0
        idx_name = 0
        for tname in sm.nodes:
            tname_to_name_dict[tname] = int(df.columns[idx_name].split('_')[-1])
            if count_lag == max_lag:
                idx_name = idx_name + 1
                count_lag = -1
            count_lag = count_lag + 1

        # Initialize the summary graph as an N x N matrix with zeros
        summary_graph = np.zeros((N, N), dtype=int)
        for ce in sm.edges:
            c = ce[0]
            c = tname_to_name_dict[c]
            e = ce[1]
            e = tname_to_name_dict[e]
            summary_graph[c, e] = 1

        # ignore the diag predictions
        for i in range(N):
            summary_graph[i, i] = 0
        res.append(
            {
                "pred": summary_graph,
                "label": example["label"],
            }
        )

    return res


def eval_by_cMLP(examples, max_lag=2):
    """
    modified from https://github.com/iancovert/Neural-GC/blob/master/cmlp_lagged_var_demo.ipynb
    """
    hidden_size = 10
    device = 'cuda'
    res = []
    for example in tqdm(examples, desc="cMLP"):
        X = example['seqs']
        X = torch.tensor(X[np.newaxis], dtype=torch.float32, device=device)
        cmlp = _cmlp.cMLP(X.shape[-1], lag=max_lag, hidden=[hidden_size]).to(device)
        train_loss_list = _cmlp.train_model_ista(cmlp, X, lam=0.1, lam_ridge=0.464159, lr=0.0005, max_iter=2000,
                                                 check_every=2000)

        # ignore the diag elements
        mask = np.eye(example["label"].shape[-1], dtype=bool)
        summary_graph = cmlp.GC().cpu().data.numpy().T
        res.append(
            {
                "pred": summary_graph[~mask],
                "label": example["label"][~mask],
            }
        )
    return res


def eval_by_cLSTM(examples):
    """
    modified from https://github.com/iancovert/Neural-GC/blob/master/clstm_lorenz_demo.ipynb
    """
    hidden_size = 10
    device = 'cuda'
    res = []
    for example in tqdm(examples, desc="cLSTM"):
        X = example["seqs"]
        X = torch.tensor(X[np.newaxis], dtype=torch.float32, device=device)
        clstm = _clstm.cLSTM(X.shape[-1], hidden=hidden_size).to(device)
        # Train with ISTA
        train_loss_list = _clstm.train_model_ista(clstm, X, context=10, lam=0.1, lam_ridge=0.010772, lr=1e-3,
                                                  max_iter=4000,
                                                  check_every=4000)
        summary_graph = clstm.GC().cpu().data.numpy().T

        # ignore the diag elements
        mask = np.eye(example["label"].shape[-1], dtype=bool)
        res.append(
            {
                "pred": summary_graph[~mask],
                "label": example["label"][~mask],
            }
        )

    return res


def save(dataset_id, method, res, root_dir=".cache/sim_data/dream3_auc_result"):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(os.path.join(root_dir, f"dataset=dream3_{dataset_id}-method={method}.pkl")):
        with open(os.path.join(root_dir, f"dataset=dream3_{dataset_id}-method={method}.pkl"), "wb") as f:
            pickle.dump(res, f)
    else:
        print(f"file {os.path.join(root_dir, f'dataset=dream3_{dataset_id}-method={method}.pkl')} already exists")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".cache/sim_data/")
    parser.add_argument("--dataset_id", type=str, default='Ecoli2')
    args = parser.parse_args()
    root_dir = args.root_dir
    dataset_id = args.dataset_id

    sims = {
        'Ecoli2': ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Ecoli2.pt',
                   '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Ecoli2.tsv'),
        'Yeast2': ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Yeast2.pt',
                   '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Yeast2.tsv'),
        'Yeast3': ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Yeast3.pt',
                   '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Yeast3.tsv')
    }

    examples = parse_examples(sims[dataset_id])
    save_dir = os.path.join(root_dir, "dream3_auc_result")
    # granger causality
    save(dataset_id, "granger_causality", eval_by_gc(examples), root_dir=save_dir)
    # mutual info
    save(dataset_id, "mutual_info", eval_by_mi(examples), root_dir=save_dir)
    # var-lingam
    save(dataset_id, "var_lingam", eval_by_lingam(examples), root_dir=save_dir)
    # pcmci+
    save(dataset_id, "pcmciplus", eval_by_pcmciplus(examples), root_dir=save_dir)
    # dynotears
    save(dataset_id, "dynotears", eval_by_dynotears(examples), root_dir=save_dir)
