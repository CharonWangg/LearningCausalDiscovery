import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pandas as pd
from causallearn.search.Granger.Granger import Granger
from causallearn.search.FCMBased import lingam
from causalnex.structure.dynotears import from_pandas_dynamic
from tigramite.independence_tests.parcorr import ParCorr
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

from external.Neural_GC.main import compute_lam_by_sweeping, compute_lam_multi_process, compute_lam_single_process
from external.SRU_for_GCI.utils.utilFuncs import loadTrainingData, loadTrueNetwork


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import warnings

warnings.filterwarnings("ignore")


def normalize(X):
    # centering
    return X - X.mean()


def parse_examples(data_paths):
    Xtrain = loadTrainingData(data_paths[0], 'cpu').float().T
    # centering and scaling
    Xtrain = normalize(Xtrain) * 50
    l, n = Xtrain.shape
    Gref = loadTrueNetwork(data_paths[1], n).T  # entry (i,j) is the causal link i -> j

    # Load data
    # Use the full length of the time series
    examples = []
    duration = 21

    # parse it into 46 snippets of sequences (the length of each one is 21 )
    for start in range(0, l, duration):
        examples.append(
            {
                "seqs": np.array(Xtrain[start:start + duration, :]),
                "label": Gref,
            }
        )
    print("data shape: ", Xtrain.shape, "graph shape: ", Gref.shape)
    # examples.append(
    #     {
    #         "seqs": np.array(Xtrain),
    #         "label": Gref,
    #     }
    # )
    return examples


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
                      verbosity=1)
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


def eval_by_dynotears_single_process(args):
    example, lambda_w, lambda_a, max_lag, max_iter = args
    X = example["seqs"]
    T, N = X.shape
    var_names = [f'X_{j}' for j in range(N)]

    df = pd.DataFrame(X, columns=var_names)
    sm = from_pandas_dynamic(df, lambda_w=lambda_w, lambda_a=lambda_a,
                             max_iter=max_iter, p=max_lag)

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

    return {
            "pred": summary_graph,
            "label": example["label"],
        }


def eval_by_dynotears(examples, dataset, multi_process=False):
    """
    modified from https://github.com/ckassaad/causal_discovery_for_time_series/blob/master/baselines/scripts_python/dynotears.py
    """
    param_config = {'Ecoli2': {'lambda_a': 0.1, 'lambda_w': 0.01, 'max_lag': 2},
        'Yeast2': {'lambda_a': 0.01, 'lambda_w': 0.01, 'max_lag': 3},
        'Yeast3': {'lambda_a': 0.01, 'lambda_w': 0.005, 'max_lag': 2},
     }
    res = []
    max_lag = param_config[dataset]['max_lag']
    lambda_w = param_config[dataset]['lambda_w']
    lambda_a = param_config[dataset]['lambda_a']
    max_iter = 1000
    if multi_process:
        with mp.Pool(8) as pool:
            args = [(example, lambda_w, lambda_a, max_lag, max_iter) for example in examples]
            res = list(tqdm(pool.imap(eval_by_dynotears_single_process, args), total=len(args)))
    else:
        for example in tqdm(examples, desc="Dynotears"):
            res.append(eval_by_dynotears_single_process((example, lambda_w, lambda_a, max_lag, max_iter)))

    return res


def eval_by_neural_granger_causality(examples, method="cmlp"):
    assert method in ["cmlp", "clstm", "sru", "esru_2LF", "esru_1LF"], f"method {method} not supported"
    method = f"compute_lam_{method}"

    # merge the examples to one input
    if method == "cmlp" or method == "clstm":
        ts = [np.stack([examples[i]["seqs"] for i in range(len(examples))], axis=0)]
    else:
        ts = [np.concatenate([examples[i]["seqs"] for i in range(len(examples))], axis=0)]
    Gref = [examples[0]["label"]]

    grefs, gests = compute_lam_multi_process(ts, Gref, method, dataset="dream3")

    res = []
    for i in range(len(grefs)):
        res.append(
            {
                "pred": gests[i],
                "label": grefs[i],
            }
        )
    return res


def check_exists(dataset_id, method, root_dir=".cache/sim_data/dream3_auc_result"):
    if os.path.exists(os.path.join(root_dir, f"dataset=dream3_{dataset_id}-method={method}.pkl")):
        print(f"file {os.path.join(root_dir, f'dataset=dream3_{dataset_id}-method={method}.pkl')} already exists")
        return True
    else:
        return False


def save(dataset_id, method, res, root_dir=".cache/sim_data/dream3_auc_result"):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(os.path.join(root_dir, f"dataset=dream3_{dataset_id}-method={method}.pkl")):
        with open(os.path.join(root_dir, f"dataset=dream3_{dataset_id}-method={method}.pkl"), "wb") as f:
            pickle.dump(res, f)
    else:
        print(f"file {os.path.join(root_dir, f'dataset=dream3_{dataset_id}-method={method}.pkl')} already exists")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

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
    # correlation
    if not check_exists(dataset_id, "correlation", root_dir=save_dir):
        save(dataset_id, "correlation", eval_by_corr(examples), root_dir=save_dir)
    # # granger causality
    if not check_exists(dataset_id, "granger_causality", root_dir=save_dir):
        save(dataset_id, "granger_causality", eval_by_gc(examples), root_dir=save_dir)
    # # mutual info
    if not check_exists(dataset_id, "mutual_info", root_dir=save_dir):
        save(dataset_id, "mutual_info", eval_by_mi(examples), root_dir=save_dir)
    # # var-lingam
    # if not check_exists(dataset_id, "var_lingam", root_dir=save_dir):
    #     save(dataset_id, "var_lingam", eval_by_lingam(examples), root_dir=save_dir)
    # cmlp
    if not check_exists(dataset_id, "cmlp", root_dir=save_dir):
        save(dataset_id, "cmlp", eval_by_neural_granger_causality(examples, method="cmlp"), root_dir=save_dir)
    # # clstm
    if not check_exists(dataset_id, "clstm", root_dir=save_dir):
        save(dataset_id, "clstm", eval_by_neural_granger_causality(examples, method="clstm"), root_dir=save_dir)
    # sru
    if not check_exists(dataset_id, "sru", root_dir=save_dir):
        save(dataset_id, "sru", eval_by_neural_granger_causality(examples, method="sru"), root_dir=save_dir)
    # esru
    if not check_exists(dataset_id, "esru", root_dir=save_dir):
        save(dataset_id, "esru", eval_by_neural_granger_causality(examples, method="esru_1LF"), root_dir=save_dir)
    # pcmci+
    if not check_exists(dataset_id, "pcmciplus", root_dir=save_dir):
        save(dataset_id, "pcmciplus", eval_by_pcmciplus(examples), root_dir=save_dir)
    #     # dynotears
    if not check_exists(dataset_id, "dynotears", root_dir=save_dir):
        save(dataset_id, "dynotears", eval_by_dynotears(examples, dataset=dataset_id, multi_process=True), root_dir=save_dir)
