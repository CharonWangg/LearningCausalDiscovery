import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pandas as pd
from causallearn.search.Granger.Granger import Granger
from causallearn.search.FCMBased import lingam
from tigramite.independence_tests.parcorr import ParCorr
from causalnex.structure.dynotears import from_pandas_dynamic
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
import torch.multiprocessing as mp
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tsaug
import pickle
import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import scipy.io as scio
from entropy_estimators import mi
from scipy.stats import pearsonr

from external.Neural_GC.main import compute_lam_by_sweeping, compute_lam_multi_process


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import warnings

warnings.filterwarnings("ignore")


def normalize(X):
    # normalize to [-1, 1]
    return (X - X.min()) / (X.max() - X.min()) * 2 - 1


def add_normalized_noise(X, scale=0.1, _seed=42):
    np.random.seed(_seed)
    return X + scale * np.random.normal(0, 1.0, X.shape) * (
                    X.max() - X.min()
                )


def parse_examples(sim, percentage, add_noise=False):
    data = scio.loadmat(sim)

    n_subjects, n_node, duration = (
        int(data["Nsubjects"]),
        int(data["Nnodes"]),
        int(data["Ntimepoints"]),
    )
    slice = [int(percentage[0] * n_subjects), int(percentage[1] * n_subjects)]
    data["ts"] = data["ts"][duration * slice[0] : duration * slice[1]]
    data["net"] = data["net"][slice[0] : slice[1]]

    if add_noise:
        # add relative scaled noise to every subject's time series
        for start in range(0, data["ts"].shape[0], duration):
            data["ts"][start : start + duration] = add_normalized_noise(
                data["ts"][start : start + duration], scale=0.1
            )
        scio.savemat(f".cache/netsim/sim{dataset_id}_noisy.mat", data)
        data = scio.loadmat(f".cache/netsim/sim{dataset_id}_noisy.mat")



    examples = []
    for i, start in enumerate(range(0, data["ts"].shape[0], duration)):
        examples.append(
            {
                "sample_id": i,
                "seqs": normalize(data["ts"][start : start + duration]),
                "label": data["net"][i],
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
            example["label"][i, i] = 0

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
        # do not need to predict the diag elements
        mask = np.eye(example["label"].shape[-1], dtype=bool)
        # take the max absolute weight value across lags
        lingam_score = np.max(np.abs(adj), axis=0).T

        # ignore the diag predictions
        for i in range(N):
            lingam_score[i, i] = 0
            example["label"][i, i] = 0

        res.append(
            {
                "pred": lingam_score,
                "label": example["label"],
            }
        )
    return res


def eval_by_corr(examples):
    res = []
    for example in tqdm(examples, desc="Correlation"):
        N = example["label"].shape[-1]
        score = np.zeros((N, N))
        for i in range(example["label"].shape[-1]):
            for j in range(example["label"].shape[-1]):
                # do not need to predict the diag elements
                if i == j:
                    continue
                corr, p_value = pearsonr(example["seqs"][:, i], example["seqs"][:, j])
                # only consider the correlation if the p_value is smaller than 0.05 (not big difference)
                if p_value >= 0.05:
                    corr = 0
                score[i, j] = corr

        for i in range(N):
            score[i, i] = 0
            example["label"][i, i] = 0

        res.append({"pred": score, "label": example["label"]})

    return res


def eval_by_mi(examples):
    res = []
    for example in tqdm(examples, desc="MI"):
        N = example["label"].shape[-1]
        score = np.zeros((N, N))
        for i in range(example["label"].shape[-1]):
            for j in range(example["label"].shape[-1]):
                # do not need to predict the diag elements
                if i == j:
                    continue
                score[i, j] = mi(example["seqs"][:, i], example["seqs"][:, j])
        for i in range(N):
            score[i, i] = 0
            example["label"][i, i] = 0

        res.append({"pred": score, "label": example["label"]})

    return res


def eval_by_pcmciplus(examples, max_lag=2):
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
            example["label"][i, i] = 0
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
            example["label"][i, i] = 0
        res.append(
            {
                "pred": summary_graph,
                "label": example["label"],
            }
        )

    return res


def eval_by_neural_granger_causality(examples, method="cmlp"):
    assert method in ["cmlp", "clstm", "sru", "esru_2LF", "esru_1LF"], f"method {method} not supported"
    method = f"compute_lam_{method}"

    ts = [examples[i]["seqs"] for i in range(len(examples))]
    Gref = [examples[i]["label"] for i in range(len(examples))]

    grefs, gests = compute_lam_multi_process(ts, Gref, method, "netsim")

    res = []
    for i in range(len(grefs)):
        res.append(
            {
                "pred": gests[i],
                "label": grefs[i],
            }
        )
    return res


def check_exists(dataset_id, method, root_dir=".cache/sim_data/netsim_auc_result"):
    post_fix = "_note=noise" if args.condition == "noise" else ""
    if os.path.exists(os.path.join(root_dir, f"dataset=netsim_{dataset_id}-method={method}{post_fix}.pkl")):
        print(f"file {os.path.join(root_dir, f'dataset=netsim_{dataset_id}-method={method}{post_fix}.pkl')} already exists")
        return True
    else:
        return False


def save(dataset_id, method, res, root_dir=".cache/sim_data/netsim_auc_result"):
    post_fix = "_note=noise" if args.condition == "noise" else ""
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if check_exists(dataset_id, method, root_dir):
        print(f"file {os.path.join(root_dir, f'dataset=netsim_{dataset_id}-method={method}{post_fix}.pkl')} already exists")
    else:
        with open(os.path.join(root_dir, f"dataset=netsim_{dataset_id}-method={method}{post_fix}.pkl"), "wb") as f:
            pickle.dump(res, f)


if __name__ == "__main__":
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".cache/sim_data/")
    parser.add_argument("--dataset_id", type=int, default=1)
    # noise-free or add noise
    parser.add_argument(
        "--condition", type=str, choices=["default", "noise"], default="default"
    )
    args = parser.parse_args()
    root_dir = args.root_dir
    dataset_id = args.dataset_id

    percentage = [0.8, 1.0]

    # all simulations have fixed sample length 200
    sim = f"{os.path.join(os.path.dirname(os.path.dirname(root_dir)), 'netsim')}/sim{dataset_id}.mat"
    save_dir = os.path.join(root_dir, "netsim_auc_result")
    examples = parse_examples(
        sim, percentage, add_noise=(args.condition == "noise")
    )
    # correlation
    if not check_exists(dataset_id, "correlation", root_dir=save_dir):
        save(dataset_id, "correlation", eval_by_corr(examples), root_dir=save_dir)
    # granger causality
    if not check_exists(dataset_id, "granger_causality", root_dir=save_dir):
        save(dataset_id, "granger_causality", eval_by_gc(examples), root_dir=save_dir)
    # mutual info
    if not check_exists(dataset_id, "mutual_info", root_dir=save_dir):
        save(dataset_id, "mutual_info", eval_by_mi(examples), root_dir=save_dir)
    # var-lingam
    if not check_exists(dataset_id, "var_lingam", root_dir=save_dir):
        save(dataset_id, "var_lingam", eval_by_lingam(examples), root_dir=save_dir)
    # pcmci+
    if not check_exists(dataset_id, "pcmciplus", root_dir=save_dir):
        save(dataset_id, "pcmciplus", eval_by_pcmciplus(examples), root_dir=save_dir)
    # dynotears
    if not check_exists(dataset_id, "dynotears", root_dir=save_dir):
        save(dataset_id, "dynotears", eval_by_dynotears(examples), root_dir=save_dir)
    # cmlp
    if not check_exists(dataset_id, "cmlp", root_dir=save_dir):
        save(dataset_id, "cmlp", eval_by_neural_granger_causality(examples, method="cmlp"), root_dir=save_dir)
    # clstm
    if not check_exists(dataset_id, "clstm", root_dir=save_dir):
        save(dataset_id, "clstm", eval_by_neural_granger_causality(examples, method="clstm"), root_dir=save_dir)
    # sru
    if not check_exists(dataset_id, "sru", root_dir=save_dir):
        save(dataset_id, "sru", eval_by_neural_granger_causality(examples, method="sru"), root_dir=save_dir)
    # esru
    if not check_exists(dataset_id, "esru", root_dir=save_dir):
        save(dataset_id, "esru", eval_by_neural_granger_causality(examples, method="esru_2LF"), root_dir=save_dir)