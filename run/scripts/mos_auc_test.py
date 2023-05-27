import os
import pickle
import random

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from causallearn.search.FCMBased import lingam
from sklearn.metrics import roc_auc_score, average_precision_score
from tigramite.independence_tests.parcorr import ParCorr
from causalnex.structure.dynotears import from_pandas_dynamic
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from entropy_estimators import mi
from multiprocessing import Pool, Manager
from copy import deepcopy
import argparse
from sklearn.linear_model import LassoCV
import time

import warnings

warnings.filterwarnings("ignore")

p_value = 0.05

def corr_test(windows):
    # Test for correlation between two sequences
    corr, p_value = pearsonr(windows[:, 0], windows[:, 1])
    # only consider the correlation if the p_value is smaller than 0.05 (not big difference)
    if p_value >= 0.05:
        corr = 0
    return corr


def mi_test(windows):
    # Test for mutual information between two sequences
    mi_score = mi(windows[:, 0], windows[:, 1])
    return mi_score


def pairwise_granger_linear_test(windows, max_lag=5):
    """
    Modified from https://github.com/tailintalent/causal/blob/master/causality/causality_unified_exp.ipynb
    """
    from sklearn.linear_model import LinearRegression

    X = windows.copy()
    Y = windows.copy()[max_lag:, 1]
    adjacency_matrix = np.zeros(max_lag)

    for lag in range(1, max_lag + 1):
        # single variable regression
        model = LinearRegression(fit_intercept=True)
        model.fit(X[max_lag - lag:-lag, 1].reshape(-1, 1), Y)
        pred = model.predict(X[max_lag - lag:-lag, 1].reshape(-1, 1))
        error_ablated = np.abs(pred - Y).mean()
        # bi-variate regression
        model = LinearRegression(fit_intercept=True)
        model.fit(X[max_lag - lag:-lag, :], Y)
        pred = model.predict(X[max_lag - lag:-lag, :])
        error = np.abs(pred - Y).mean()
        # calculate the granger causality score
        adjacency_matrix[lag - 1] = np.log(error_ablated / error)

    return adjacency_matrix.max()


def var_lingam_test(windows, max_lag=5):
    model = lingam.VARLiNGAM(lags=max_lag)
    model.fit(windows)
    adj = model.adjacency_matrices_  # shape: (max_lag, n_features, n_features)
    # take the max absolute weight value across lags
    lingam_score = np.max(np.abs(adj[:, 1, 0]))
    return lingam_score


def dynotears_test(windows, max_lag=5):
    time_start = time.time()
    T, N = windows.shape
    var_names = [f'X_{j}' for j in range(N)]

    df = pd.DataFrame(windows, columns=var_names)
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

    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    return summary_graph


def pcmciplus_test(windows, max_lag=5):
    time_start = time.time()
    T, N = windows.shape
    var_names = [f'X_{j}' for j in range(N)]
    df = pp.DataFrame(windows, var_names=var_names)
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

    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    return summary_graph


class AUCTest:
    def __init__(
            self,
            mat,
            df,
            methods="all",
            maxlag=5,
            n_jobs=1,
            root_dir=".cache/results",
    ):
        self.mat = mat
        self.df = df
        self.maxlag = maxlag
        self.n_jobs = n_jobs
        self.methods = {
            "corr": corr_test,
            "mi": mi_test,
            "gc": pairwise_granger_linear_test,
            "lingam": var_lingam_test,
            # "pcmciplus": pcmciplus_test,
            # "dynotears": dynotears_test,
        }
        if methods != "all":
            self.methods = {k: v for k, v in self.methods.items() if k in methods}
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)


    def run_all_methods(self, save_cfg=dict()):
        print(
            f'Test: all methods {self.methods.keys()} | Game: {save_cfg["game"]} | Sequence Length: {self.mat.shape[1]} '
            f'| Transform: {save_cfg["note"]} | Seed: {save_cfg["seed"]}'
        )
        for method, _ in self.methods.items():
            temp_cfg = {"method": method}
            temp_cfg.update(save_cfg)
            if os.path.exists(self.format_results(temp_cfg)):
                print(
                    f"File already exists: {self.format_results(temp_cfg)}, skipping..."
                )
            else:
                print(f"Running {method}...")
                self.run(method, save_cfg=deepcopy(save_cfg))

    def run(self, method="corr", save_cfg=dict()):
        assert method in self.methods.keys(), f"{method} is not supported"
        graph_based_methods = ["dynotears", "pcmciplus"]
        pairwise_methods = ["corr", "mi", "gc", "lingam"]
        if self.n_jobs == 1 or method in graph_based_methods:
            results = self.single_process_test_for_graph(
                {
                    "df": self.df,
                    "method": method,
                    "mat": self.mat,
                }
            )
            results = {method: results}
        else:
            dfs = [self.df.iloc[i:: self.n_jobs] for i in range(self.n_jobs)]
            p = Pool(self.n_jobs)
            manager = Manager()
            mat = manager.list()
            mat.extend([seq for seq in self.mat])
            results = []

            for df in dfs:
                temp_result = p.apply_async(
                    self.single_process_test,
                    args=(
                        {
                            "df": df,
                            "mat": mat,
                            "method": method,
                        },
                    ),
                )
                results.append(temp_result)

            p.close()  # close pool
            p.join()  # wait for all processes to finish
            print("All subprocesses done.")
            results = [result.get() for result in results]

            results = {
                method: {
                    "pred": sum([result["pred"] for result in results], []),
                    "label": sum([result["label"] for result in results], []),
                }
            }
        print(
            "method test auc: ",
            roc_auc_score(results[method]["label"], np.abs(results[method]["pred"])),
            "method test ap: ",
            average_precision_score(results[method]["label"], np.abs(results[method]["pred"])),
        )

        save_cfg["method"] = method
        self.save(results, save_cfg=save_cfg)

    def single_process_test_for_graph(self, kwargs):
        df = kwargs["df"]
        method, method_imp = kwargs["method"], self.methods[kwargs["method"]]
        mat = kwargs["mat"]

        # use the entire mat to do graph-based causal discovery
        transistor_idxs = df["transistor_1"].unique().tolist()
        windows = np.stack([mat[idx] for idx in transistor_idxs], axis=0).T
        adj_mat = np.zeros((windows.shape[0], windows.shape[0]))
        for i, row in df.iterrows():
            adj_mat[
                transistor_idxs.index(row["transistor_1"]),
                transistor_idxs.index(row["transistor_2"]),
            ] = row["label"]

        pred = method_imp(windows, max_lag=self.maxlag)

        # ignore the diagonal
        mask = np.ones(adj_mat.shape, dtype=bool)
        # flatten the dict into a list
        results = {"pred": pred[~mask].reshape(-1).tolist(),
                   "label": adj_mat[~mask].reshape(-1).tolist()}
        return results

    def single_process_test(self, kwargs):
        df = kwargs["df"]
        method, method_imp = kwargs["method"], self.methods[kwargs["method"]]
        mat = kwargs["mat"]

        _results = {}

        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            # if method is symmetrical, only need to test one direction
            if method in ["corr", "mi"]:
                if (int(row["transistor_2"]), int(row["transistor_1"])) in _results:
                    _results[(int(row["transistor_1"]), int(row["transistor_2"]))] = {
                        "pred": _results[(int(row["transistor_2"]), int(row["transistor_1"]))]["pred"],
                        "label": row["label"],
                    }
                    continue

            windows = np.stack(
                (
                    mat[row["transistor_1"]],
                    mat[row["transistor_2"]],
                ),
                axis=-1,
            )
            _results[(int(row["transistor_1"]), int(row["transistor_2"]))] = {
                "pred": method_imp(windows, max_lag=self.maxlag) if method in ["lingam", "gc"] else method_imp(windows),
                "label": row["label"],
            }

        # flatten the dict into a list
        results = {"pred": [v["pred"] for _, v in _results.items()],
                   "label": [v["label"] for _, v in _results.items()]}
        return results

    def format_results(self, save_cfg=dict()):
        game, window, method, seed, note = (
            save_cfg["game"],
            save_cfg["window"],
            save_cfg["method"],
            save_cfg["seed"],
            save_cfg.get("note", "default"),
        )
        if not os.path.exists(os.path.join(self.root_dir, "mos_auc_result")):
            os.makedirs(os.path.join(self.root_dir, "mos_auc_result"))
        return os.path.join(
            self.root_dir,
            "mos_auc_result",
            f"game={game}-window={window[0]}_{window[1]}-method={method}-seed={seed}-note={note}.pkl",
        )

    def save(self, result, save_cfg=dict()):
        formatted_dir = self.format_results(save_cfg)
        if not os.path.exists(formatted_dir):
            with open(formatted_dir, "wb") as f:
                pickle.dump(result, f)
            print(f"{formatted_dir} saved")
        else:
            print(f"{formatted_dir} already exists")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".cache/sim_data/")
    parser.add_argument("--game", type=str, choices=["DonkeyKong", "Pitfall", "SpaceInvaders", "All"],
                        default="DonkeyKong")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--n_jobs", type=int, default=1)

    args = parser.parse_args()
    print(args)
    interval = 1
    maxlag = 5
    n_jobs = args.n_jobs
    games = [args.game] if args.game != "All" else ["DonkeyKong", "Pitfall", "SpaceInvaders"]
    seed = 42
    selected_methods = "all"
    debug = args.debug
    root_dir = args.root_dir

    test_windows = [(0, 128), (128, 256), (384, 512), (512, 640), (640, 768)]

    # no noise
    for test_window in test_windows:
        for game in games:
            random.seed(seed)
            np.random.seed(seed)
            mat = np.load(
                os.path.join(root_dir,
                             f"{game}/HR/window_{test_window[0]}_{test_window[1]}/Regular_3510_step_128_rec_30_window_{test_window[0]}_{test_window[1]}.npy"),
                mmap_mode="r",
            ).astype(np.float32)[:, ::interval]
            # add very small noise to avoid repeated and constant states because of downsampling
            mat += np.random.normal(0, 1e-3, mat.shape)
            df = pd.read_csv(
                os.path.join(
                    root_dir, f"DonkeyKong/HR/window_{test_window[0]}_{test_window[1]}/csv/fold_seed_{seed}/test_ds_1.0_unique_True.csv"
                )
            )
            if debug:
                df = df.iloc[:1000]
            tester = AUCTest(
                mat,
                df,
                methods=selected_methods,
                maxlag=maxlag,
                n_jobs=n_jobs,
                root_dir=root_dir,
            )
            tester.run_all_methods(
                save_cfg={"game": game, "window": test_window, "seed": seed, "note": "default"}
            )

    # # under noise std 0.03, 0.05, 0.1
    transforms = [0.03, 0.05, 0.1]
    games = ["DonkeyKong"]
    for test_window in test_windows:
        for game in games:
            random.seed(seed)
            np.random.seed(seed)
            for transform in transforms:
                mat = np.load(
                    os.path.join(
                        root_dir, f"{game}/HR/window_{test_window[0]}_{test_window[1]}/Regular_3510_step_128_rec_30_window_{test_window[0]}_{test_window[1]}_Noise_{transform}.npy"
                    ),
                    mmap_mode="r",
                ).astype(np.float32)[:, ::interval]
                df = pd.read_csv(
                    os.path.join(
                                        root_dir, f"{game}/HR/window_{test_window[0]}_{test_window[1]}/csv/fold_seed_{seed}/test_ds_1.0_unique_True.csv"
                                    )
                )
                if debug:
                    df = df.iloc[:1000]
                tester = AUCTest(
                    mat,
                    df,
                    methods=selected_methods,
                    maxlag=maxlag,
                    n_jobs=n_jobs,
                    root_dir=root_dir,
                )
                tester.run_all_methods(
                    save_cfg={"game": game, "window": test_window, "seed": seed, "note": f"{transform}noise"}
                )
