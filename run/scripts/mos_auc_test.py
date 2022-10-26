import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from causallearn.search.FCMBased import lingam
from sklearn.metrics import roc_auc_score
from entropy_estimators import mi
from tsaug import AddNoise
from multiprocessing import Pool, Manager
from copy import deepcopy
import argparse
from sklearn.linear_model import LassoCV

import warnings
warnings.filterwarnings("ignore")


def get_noiser(scale=0.1, seed=42, normalize=False):
    return AddNoise(scale=scale, seed=seed, normalize=normalize).augment


def corr_test(windows):
    # Test for correlation between two sequences
    # TODO: calculate only the half symmetric part to save time
    corr, p_value = pearsonr(windows[:, 0], windows[:, 1])
    # only consider the correlation if the p_value is smaller than 0.05 (not big difference)
    # if p_value >= 0.05:
    #     corr = 0
    return corr


def mi_test(windows):
    # Test for mutual information between two sequences
    # TODO: calculate only the half symmetric part to save time
    mi_score = mi(windows[:, 0], windows[:, 1])
    return mi_score


def pairwise_granger_lasso_test(windows, max_lag=5, cv=5, seed=42):
    # lasso granger causality test between two sequences with a fixed lag
    # windows: (n_samples, n_features(2)), only test if 0 -> 1
    Y = windows[max_lag:]
    X = np.hstack([windows[maxlag - lag:-lag] for lag in range(1, maxlag + 1)])

    lasso_cv = LassoCV(cv=cv, random_state=seed)
    # consider only the second variable as the 'effect' variable
    lasso_cv.fit(X, Y[:, 1])
    # calculate the largest absolute weight value of each edge across lags
    coeff = np.max(np.abs(lasso_cv.coef_[::2]))
    return coeff


def gc_test(windows):
    # granger causality test between two sequences
    coeff = pairwise_granger_lasso_test(windows)
    return coeff


def lingam_test(windows, seed=42):
    model = lingam.ICALiNGAM(random_state=seed)
    model.fit(windows)
    lingam_score = np.where(model.adjacency_matrix_!=0, 1, 0)[1, 0]
    return lingam_score


class AUCTest:
    def __init__(self, mat, df, methods='all', transform=None, maxlag=5, n_jobs=1, root_dir='.cache/results'):
        self.mat = mat
        self.df = df
        self.transform = transform
        self.maxlag = maxlag
        self.n_jobs = n_jobs
        self.methods = {
            'corr': corr_test,
            'mi': mi_test,
            'gc': gc_test,
            'lingam': lingam_test,
        }
        if methods != 'all':
            self.methods = {k: v for k, v in self.methods.items() if k in methods}
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def run_all_methods(self, save_cfg=dict()):
        print(f'Test: all methods {self.methods.keys()} | Game: {save_cfg["game"]} | Transform: {save_cfg["note"]} | Seed: {save_cfg["seed"]}')
        for method, _ in self.methods.items():
            temp_cfg = {'method': method}
            temp_cfg.update(save_cfg)
            if os.path.exists(self.format_results(temp_cfg)):
                print(f'File already exists: {self.format_results(temp_cfg)}, skipping...')
            else:
                print(f'Running {method}...')
                self.run(method, save_cfg=deepcopy(save_cfg))

    def run(self, method='corr', save_cfg=dict()):
        assert method in self.methods.keys(), f'{method} is not supported'
        if self.n_jobs == 1:
            results = self.single_process_test({'df': self.df, 'method': method, 'mat': self.mat, 'transform': self.transform})
            results = {method: results}
        else:
            dfs = [self.df.iloc[i::self.n_jobs] for i in range(self.n_jobs)]
            p = Pool(self.n_jobs)
            manager = Manager()
            mat = manager.list()
            mat.extend([seq for seq in self.mat])
            results = []
            for df in dfs:
                temp_result = p.apply_async(self.single_process_test, args=({'df': df, 'mat': mat,
                                                        'transform': self.transform, 'method': method},))
                results.append(temp_result)
            p.close()  # close pool
            p.join()  # wait for all processes to finish
            print('All subprocesses done.')
            results = [result.get() for result in results]

            results = {method: {'pred': sum([result['pred'] for result in results], []),
                                'label': sum([result['label'] for result in results], [])}}
        print('method test auc: ', roc_auc_score(results[method]['label'], results[method]['pred']))

        save_cfg['method'] = method
        self.save(results, save_cfg=save_cfg)

    def single_process_test(self, kwargs):
        df = kwargs['df']
        method, method_imp = kwargs['method'], self.methods[kwargs['method']]
        mat = kwargs['mat']
        transform = kwargs['transform']

        results = {'pred': [], 'label': []}

        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            if transform is None:
                # get all transistor id in cause and effect
                windows = np.stack((mat[row['transistor_1']],
                                    mat[row['transistor_2']]), axis=-1).astype(np.int)
            else:
                windows = np.stack((transform(mat[row['transistor_1']]),
                                    transform(mat[row['transistor_2']])), axis=-1)

            results['pred'].append(method_imp(windows))

        results['label'] = df['label'].tolist()
        return results

    def format_results(self, save_cfg=dict()):
        game, method, seed, note = save_cfg['game'], save_cfg['method'], save_cfg['seed'], save_cfg.get('note', 'default')
        if not os.path.exists(os.path.join(self.root_dir, 'mos_auc_result')):
            os.makedirs(os.path.join(self.root_dir, 'mos_auc_result'))
        return os.path.join(self.root_dir, 'mos_auc_result', f'game={game}-method={method}-seed={seed}-note={note}.pkl')

    def save(self, result, save_cfg=dict()):
        formatted_dir = self.format_results(save_cfg)
        if not os.path.exists(formatted_dir):
            with open(formatted_dir, 'wb') as f:
                pickle.dump(result, f)
            print(f'{formatted_dir} saved')
        else:
            print(f'{formatted_dir} already exists')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='.cache/sim_data/')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    interval = 10
    window_size = 512
    maxlag = 5
    n_jobs = args.n_jobs
    games = ['DonkeyKong']
    seeds = [42, 7, 12, 1207, 3057]
    selected_methods = 'all'
    debug = args.debug
    root_dir = args.root_dir

    # no noise
    for game in games:
        for seed in seeds:
            mat = np.load(os.path.join(root_dir, f'{game}/HR/Regular_3510_step_256_rec_2e3.npy'), mmap_mode='r').astype(np.float32)[:, ::interval]
            df = pd.read_csv(os.path.join(root_dir, f'{game}/csv/fold_seed_{seed}/test_sim_grouped.csv'))
            if debug:
                df = df.iloc[:1000]
            tester = AUCTest(mat, df, methods=selected_methods, transform=None, maxlag=maxlag, n_jobs=n_jobs, root_dir=root_dir)
            tester.run_all_methods(save_cfg={'game': game, 'seed': seed, 'note': 'default'})

    # under noise std 0.1, 0.3, 0.5
    transforms = ['0.1noise', '0.3noise', '0.5noise']
    game = ['DonkeyKong']
    for game in games:
        for seed in seeds:
            for transform in transforms:
                mat = np.load(os.path.join(root_dir, f'{game}/HR/Regular_3510_step_256_rec_2e3.npy'),
                              mmap_mode='r').astype(np.float32)[:, ::interval]
                df = pd.read_csv(os.path.join(root_dir, f'{game}/csv/fold_seed_{seed}/test_sim_grouped.csv'))
                if debug:
                    df = df.iloc[:1000]
                tester = AUCTest(mat, df, methods=selected_methods, transform=get_noiser(float(transform.strip('noise')), seed, normalize=False),
                                 maxlag=maxlag, n_jobs=n_jobs, root_dir=root_dir)
                tester.run_all_methods(save_cfg={'game': game, 'seed': seed, 'note': transform})