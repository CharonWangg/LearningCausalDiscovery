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
from entropy_estimators import mi
from multiprocessing import Pool, Manager
from copy import deepcopy
import argparse
from sklearn.linear_model import LassoCV

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


# def pcmci_test():

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
        if self.n_jobs == 1:
            results = self.single_process_test(
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
        game, method, seed, note = (
            save_cfg["game"],
            save_cfg["method"],
            save_cfg["seed"],
            save_cfg.get("note", "default"),
        )
        if not os.path.exists(os.path.join(self.root_dir, "mos_auc_result")):
            os.makedirs(os.path.join(self.root_dir, "mos_auc_result"))
        return os.path.join(
            self.root_dir,
            "mos_auc_result",
            f"game={game}-method={method}-seed={seed}-note={note}.pkl",
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
    interval = 100
    maxlag = 5
    n_jobs = args.n_jobs
    games = [args.game] if args.game != "All" else ["DonkeyKong", "Pitfall", "SpaceInvaders"]
    seeds = [42, 1, 2, 3, 4]
    selected_methods = "all"
    debug = args.debug
    root_dir = args.root_dir

    # no noise
    # for game in games:
    #     for seed in seeds:
    #         random.seed(seed)
    #         np.random.seed(seed)
    #         mat = np.load(
    #             os.path.join(root_dir,
    #                          f"{game}/HR/window_768_1024/Regular_3510_step_256_rec_400_window_768_1024.npy"),
    #             mmap_mode="r",
    #         ).astype(np.float32)[:, ::interval]
    #         # add very small noise to avoid repeated and constant states because of downsampling
    #         mat += np.random.normal(0, 1e-3, mat.shape)
    #         df = pd.read_csv(
    #             os.path.join(
    #                 root_dir, f"{game}/HR/window_768_1024/csv/fold_seed_{seed}/test.csv"
    #             )
    #         )
    #         if debug:
    #             df = df.iloc[:1000]
    #         tester = AUCTest(
    #             mat,
    #             df,
    #             methods=selected_methods,
    #             maxlag=maxlag,
    #             n_jobs=n_jobs,
    #             root_dir=root_dir,
    #         )
    #         tester.run_all_methods(
    #             save_cfg={"game": game, "seed": seed, "note": "default"}
    #         )

    # # under noise std 0.1, 0.3, 0.5
    transforms = [0.1, 0.3, 0.5]
    games = ["DonkeyKong"]
    for game in games:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            for transform in transforms:
                mat = np.load(
                    os.path.join(
                        root_dir, f"{game}/HR/window_768_1024/Regular_3510_step_256_rec_400_window_768_1024_Noise_{transform}.npy"
                    ),
                    mmap_mode="r",
                ).astype(np.float32)[:, ::interval]
                df = pd.read_csv(
                    os.path.join(
                        root_dir, f"{game}/HR/window_768_1024/csv/fold_seed_{seed}/test.csv"
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
                    save_cfg={"game": game, "seed": seed, "note": f"{transform}noise"}
                )
