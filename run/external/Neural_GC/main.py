import torch
import numpy as np
import scipy.io as scio
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import multiprocessing as mp
from tqdm import tqdm
from sklearn.metrics import auc

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys

sys.path.append('/project/lcd_v2/run/external')

from Neural_GC.models import cmlp as _cmlp
from Neural_GC.models import clstm as _clstm

from SRU_for_GCI.models.sru import SRU, trainSRU
from SRU_for_GCI.models.esru_1LF import eSRU_1LF, train_eSRU_1LF
from SRU_for_GCI.models.esru_2LF import eSRU_2LF, train_eSRU_2LF
from SRU_for_GCI.utils.utilFuncs import env_config, loadTrainingData, loadTrueNetwork, getCausalNodes, count_parameters, \
    getGeneTrainingData

hidden_size = 10
device = 'cuda'

param_config = {
    "netsim": {
        "cmlp": {
            "hidden_size": 10,
            "lam_ridge": 0.464159,
            "lr": 5e-4,
            "max_iter": 2000,
            "lam": [0.1, 3.162],
            "lam_output": 0,
            "max_lag": 5,

        },
        "clstm": {
            "hidden_size": 10,
            "lam_ridge": 0.010772,
            "lr": 1e-3,
            "max_iter": 4000,
            "lam": [0.1, 3.162],
            "lam_output": 0,
        },
        "sru": {
            "A": [0.0, 0.01, 0.1, 0.99],
            "dim_iid_stats": 10,
            "dim_rec_stats": 10,
            "dim_final_stats": 10,
            "dim_rec_stats_feedback": 10,
            "batchSize": 10,
            "numTotalSamples": 200,
            "lam": [0.1, 3.162],
            "lam_ridge": 0.464159,
            "lam_output": 0,
            "lr": 1e-3,
            "max_iter": 2000,
        },
        "esru": {
            "A": [0.0, 0.01, 0.1, 0.99],
            "dim_iid_stats": 10,
            "dim_rec_stats": 10,
            "dim_final_stats": 10,
            "dim_rec_stats_feedback": 10,
            "batchSize": 10,
            "numTotalSamples": 200,
            "lam": [0.1, 3.162],
            "lam_ridge": 0.232,
            "lam_output": 0.005,
            "lr": 1e-3,
            "max_iter": 2000,

        },
    },
    "dream3": {
        "cmlp": {
            "hidden_size": 5,
            "lam_ridge": 5,
            "lr": 5e-4,
            "lam": [0.1, 100],
            "lam_output": 0,
            "max_iter": 2000,
            "max_lag": 2,

        },
        "clstm": {
            "hidden_size": 10,
            "lam_ridge": 5,
            "lr": 5e-4,
            "lam": [0.1, 17.52],
            "lam_output": 0,
            "max_iter": 2000,
        },
        "sru": {
            "A": [0.0, 0.01, 0.1, 0.5, 0.99],
            "dim_iid_stats": 10,
            "dim_rec_stats": 10,
            "dim_final_stats": 10,
            "dim_rec_stats_feedback": 10,
            "batchSize": 21,
            "numTotalSamples": 966,
            "lam": [0.01, 1.0],
            "lam_ridge": 0.2,
            "lam_output": 0,
            "lr": 5e-3,
            "max_iter": 1000,
        },
        "esru": {
            "A": [0.0, 0.01, 0.1, 0.5, 0.99],
            "dim_iid_stats": 10,
            "dim_rec_stats": 10,
            "dim_final_stats": 10,
            "dim_rec_stats_feedback": 10,
            "batchSize": 21,
            "numTotalSamples": 966,
            "lam": [0.1, 3.162],
            "lam_ridge": 0.1,
            "lam_output": 1,
            "lr": 1e-3,
            "max_iter": 2000,
        },
    }
}

def compute_lam_by_sweeping_v2(ts, Gref, method, dataset):
    method_name = method.split('_')[2]
    lams = param_config[dataset][method_name]['lam']
    lams = np.linspace(lams[0], lams[1], 10, endpoint=True)

    score_auroc, score_auprc = [], []
    tpr_lists, fpr_lists, precision_lists, recall_lists = [], [], [], []
    for subj, (_ts, _gref) in tqdm(enumerate(zip(ts, Gref)), total=len(ts)):
        # set number of processes
        with mp.Pool(10) as pool:
            stats = pool.map(compute_lam_on_single_subject, [(_ts, _gref, lam, method, dataset) for lam in lams])

        tpr_list = np.array([stat[0] for stat in stats])
        fpr_list = np.array([stat[1] for stat in stats])
        precision_list = np.array([stat[2] for stat in stats])
        recall_list = np.array([stat[0] for stat in stats])

        # get sorted indices for ROC (ascending order) and PR curve (descending order)
        roc_sorted_indices = np.argsort(fpr_list)
        pr_sorted_indices = np.argsort(recall_list)[::-1]

        # sort the data
        fpr_list_sorted = fpr_list[roc_sorted_indices]
        tpr_list_sorted = tpr_list[roc_sorted_indices]

        recall_list_sorted = recall_list[pr_sorted_indices]
        precision_list_sorted = precision_list[pr_sorted_indices]

        # calculate AUROC and AUPRC
        auroc = auc(fpr_list_sorted, tpr_list_sorted)
        auprc = auc(recall_list_sorted, precision_list_sorted)

        score_auroc.append(auroc)
        score_auprc.append(auprc)

        tpr_lists.append(tpr_list)
        fpr_lists.append(fpr_list)
        precision_lists.append(precision_list)
        recall_lists.append(recall_list)
        stats_dict = {
            'tpr': tpr_lists,
            'fpr': fpr_lists,
            'precision': precision_lists,
            'recall': recall_lists,
        }

    return np.mean(score_auroc), np.mean(score_auprc), stats_dict


def compute_lam_by_sweeping(ts, Gref, method, dataset):
    method_name = method.split('_')[2]
    lams = param_config[dataset][method_name]['lam']
    lams = np.linspace(lams[0], lams[1], 10, endpoint=True)

    score_auroc, score_auprc = [], []
    for subj, (_ts, _gref) in tqdm(enumerate(zip(ts, Gref)), total=len(ts)):
        # set number of processes
        with mp.Pool(10) as pool:
            stats = pool.map(compute_lam_on_single_subject, [(_ts, _gref, lam, method, dataset) for lam in lams])

        tpr_list = [stat[0] for stat in stats]
        fpr_list = [stat[1] for stat in stats]
        precision_list = [stat[2] for stat in stats]
        recall_list = [stat[0] for stat in stats]

        sorted_indices = np.argsort(fpr_list)
        fpr_list, tpr_list = np.array(fpr_list)[sorted_indices], np.array(tpr_list)[sorted_indices]

        sorted_indices = np.argsort(precision_list)
        recall_list, precision_list = np.array(recall_list)[sorted_indices], np.array(precision_list)[sorted_indices]

        # calculate auroc
        fpr_list = np.insert(fpr_list, 0, 0)
        tpr_list = np.insert(tpr_list, 0, 0)
        fpr_list = np.append(fpr_list, 1)
        tpr_list = np.append(tpr_list, 1)

        n = len(fpr_list)
        auroc = 0
        for ii in range(n - 1):
            h = fpr_list[ii + 1] - fpr_list[ii]
            b1 = tpr_list[ii]
            b2 = tpr_list[ii + 1]
            trapezoid_area = 0.5 * h * (b1 + b2)
            auroc = auroc + trapezoid_area

        # calculate auprc
        if (precision_list[0] > 0):
            precision_list = np.insert(precision_list, 0, 0)
            recall_list = np.insert(recall_list, 0, 1)
        if (precision_list[len(precision_list) - 1] < 1):
            precision_list = np.append(precision_list, 1)
            recall_list = np.append(recall_list, 0)

        n = len(precision_list)
        auprc = 0
        for ii in range(n - 1):
            h = precision_list[ii + 1] - precision_list[ii]
            b1 = recall_list[ii]
            b2 = recall_list[ii + 1]
            trapezoid_area = 0.5 * h * (b1 + b2)
            auprc = auprc + trapezoid_area

        score_auroc.append(auroc)
        score_auprc.append(auprc)

    return np.mean(score_auroc), np.mean(score_auprc)


def compute_lam_cmlp(args):
    lam, X, _gref, dataset = args
    X = torch.tensor(X, dtype=torch.float32, device=device)
    lam_ridge = param_config[dataset]['cmlp']['lam_ridge']
    lr = param_config[dataset]['cmlp']['lr']
    max_iter = param_config[dataset]['cmlp']['max_iter']
    hidden_size = param_config[dataset]['cmlp']['hidden_size']
    max_lag = param_config[dataset]['cmlp']['max_lag']
    if dataset == 'dream3':
        duration = 21
        X = X.reshape(-1, duration, X.shape[-1])
        print(X.shape)
    cmlp = _cmlp.cMLP(X.shape[-1], lag=max_lag, hidden=[hidden_size]).to(device)
    train_loss_list = _cmlp.train_model_ista(cmlp, X, lam=lam, lam_ridge=lam_ridge, lr=lr, max_iter=max_iter,
                                             check_every=100, verbose=2)

    Gest = cmlp.GC(threshold=True).cpu().data.numpy().T

    return lam, Gest


def compute_lam_clstm(args):
    lam, X, _gref, dataset = args
    lam_ridge = param_config[dataset]['clstm']['lam_ridge']
    lr = param_config[dataset]['clstm']['lr']
    max_iter = param_config[dataset]['clstm']['max_iter']
    hidden_size = param_config[dataset]['clstm']['hidden_size']
    X = torch.tensor(X, dtype=torch.float32, device=device)
    if dataset == 'dream3':
        duration = 21
        X = X.reshape(-1, duration, X.shape[-1])
        print(X.shape)
    clstm = _clstm.cLSTM(X.shape[-1], hidden=hidden_size).to(device)
    train_loss_list = _clstm.train_model_ista(clstm, X, context=10, lam=lam, lam_ridge=lam_ridge, lr=lr, max_iter=max_iter,
                                              check_every=100, verbose=2)

    Gest = clstm.GC(threshold=False).cpu().data.numpy().T

    return lam, Gest


def compute_lam_sru(args):
    lam, X, _gref, dataset = args
    N = X.shape[-1]
    X = torch.tensor(X.T, dtype=torch.float32, device=device).squeeze()

    A = param_config[dataset]['sru']['A']
    dim_iid_stats = param_config[dataset]['sru']['dim_iid_stats']
    dim_rec_stats = param_config[dataset]['sru']['dim_rec_stats']
    dim_final_stats = param_config[dataset]['sru']['dim_final_stats']
    dim_rec_stats_feedback = param_config[dataset]['sru']['dim_rec_stats_feedback']
    batchSize = param_config[dataset]['sru']['batchSize']
    numTotalSamples = param_config[dataset]['sru']['numTotalSamples']
    numBatches = int(numTotalSamples / batchSize)
    if dataset == "netsim":
        blk_size = int(batchSize / 2)
    else:
        blk_size = batchSize
    max_iter = param_config[dataset]['sru']['max_iter']
    lam_ridge = param_config[dataset]['sru']['lam_ridge']
    lr = param_config[dataset]['sru']['lr']

    Gest = torch.zeros(N, N, requires_grad=False)
    for predictedNode in range(N):
        model = SRU(N, 1, dim_iid_stats, dim_rec_stats, dim_rec_stats_feedback,
                    dim_final_stats, A, device)
        model.to(device)  # shift to CPU/GPU memory
        model, lossVec = trainSRU(model, X, device, numBatches, batchSize, blk_size, predictedNode,
                                  max_iter=max_iter,
                                  lambda1=lam_ridge, lambda2=lam, lr=lr, lr_gamma=0.99, lr_update_gap=4,
                                  staggerTrainWin=1, stoppingThresh=1e-5, verbose=1)
        Gest.data[predictedNode, :] = torch.norm(model.lin_xr2phi.weight.data[:, :N], p=2, dim=0)

    Gest = (Gest > 0).to(torch.int)

    return lam, Gest.cpu().data.numpy().T


def compute_lam_esru_1LF(args):
    lam, X, _gref, dataset = args
    N = X.shape[-1]
    duration = 21
    X = torch.tensor(X.T, dtype=torch.float32, device=device).squeeze()

    A = param_config[dataset]["esru"]["A"]
    dim_iid_stats = param_config[dataset]["esru"]["dim_iid_stats"]
    dim_rec_stats = param_config[dataset]["esru"]["dim_rec_stats"]
    dim_final_stats = param_config[dataset]["esru"]["dim_final_stats"]
    dim_rec_stats_feedback = param_config[dataset]["esru"]["dim_rec_stats_feedback"]
    batchSize = param_config[dataset]["esru"]["batchSize"]
    if dataset == "netsim":
        blk_size = int(batchSize / 2)
    else:
        blk_size = batchSize
    numTotalSamples = param_config[dataset]["esru"]["numTotalSamples"]
    numBatches = int(numTotalSamples / batchSize)
    max_iter = param_config[dataset]["esru"]["max_iter"]
    lam_ridge = param_config[dataset]["esru"]["lam_ridge"]
    lam_output = param_config[dataset]["esru"]["lam_output"]
    lr = param_config[dataset]["esru"]["lr"]

    Gest = torch.zeros(N, N, requires_grad=False)
    for predictedNode in range(N):
        model = eSRU_1LF(N, 1, dim_iid_stats, dim_rec_stats, dim_rec_stats_feedback,
                         dim_final_stats, A, device)
        model.to(device)  # shift to CPU/GPU memory
        model, lossVec = train_eSRU_1LF(model, X, device, numBatches, batchSize, blk_size, predictedNode,
                                        max_iter=max_iter,
                                        lambda1=lam_ridge,
                                        lambda2=lam, lambda3=lam_output, lr=lr, lr_gamma=0.99,
                                        lr_update_gap=4,
                                        staggerTrainWin=1, stoppingThresh=1e-5, verbose=1)
        Gest.data[predictedNode, :] = torch.norm(model.lin_xr2phi.weight.data[:, :N], p=2, dim=0)

    Gest = (Gest > 0).to(torch.int)

    return lam, Gest.cpu().data.numpy().T


def compute_lam_esru_2LF(args):
    lam, X, _gref, dataset = args
    N = X.shape[-1]
    X = torch.tensor(X.T, dtype=torch.float32, device=device).squeeze()

    A = param_config[dataset]["esru"]["A"]
    dim_iid_stats = param_config[dataset]["esru"]["dim_iid_stats"]
    dim_rec_stats = param_config[dataset]["esru"]["dim_rec_stats"]
    dim_final_stats = param_config[dataset]["esru"]["dim_final_stats"]
    dim_rec_stats_feedback = param_config[dataset]["esru"]["dim_rec_stats_feedback"]
    batchSize = param_config[dataset]["esru"]["batchSize"]
    if dataset == "netsim":
        blk_size = int(batchSize / 2)
    else:
        blk_size = batchSize
    numTotalSamples = param_config[dataset]["esru"]["numTotalSamples"]
    numBatches = int(numTotalSamples / batchSize)
    max_iter = param_config[dataset]["esru"]["max_iter"]
    lam_ridge = param_config[dataset]["esru"]["lam_ridge"]
    lam_output = param_config[dataset]["esru"]["lam_output"]
    lr = param_config[dataset]["esru"]["lr"]

    Gest = torch.zeros(N, N, requires_grad=False)
    for predictedNode in range(N):
        model = eSRU_2LF(N, 1, dim_iid_stats, dim_rec_stats, dim_rec_stats_feedback,
                         dim_final_stats, A, device)
        model.to(device)  # shift to CPU/GPU memory
        model, lossVec = train_eSRU_2LF(model, X, device, numBatches, batchSize, blk_size, predictedNode,
                                        max_iter=max_iter,
                                        lambda1=lam_ridge,
                                        lambda2=lam, lambda3=lam_output, lr=lr, lr_gamma=0.99,
                                        lr_update_gap=4,
                                        staggerTrainWin=1, stoppingThresh=1e-5, verbose=0)
        Gest.data[predictedNode, :] = torch.norm(model.lin_xr2phi.weight.data[:, :N], p=2, dim=0)

    Gest = (Gest > 0).to(torch.int)

    return lam, Gest.cpu().data.numpy().T


def compute_lam_on_single_subject(args):
    _ts, _gref, lam, method, dataset = args
    method = getattr(sys.modules[__name__], method)

    X = torch.tensor(_ts[np.newaxis], dtype=torch.float32, device=device).cpu().numpy()
    _, gest = method((lam, X, _gref, dataset))

    # ignore the diagonal elements
    mask = np.eye(gest.shape[-1], dtype=bool)
    gest = gest[~mask]
    _gref = _gref[~mask]

    gest = gest.flatten()
    gref = np.where(_gref.flatten() != 0, 1, 0)

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(gest)):
        if gest[i] == 1 and gref[i] == 1:
            TP += 1
        elif gest[i] == 1 and gref[i] == 0:
            FP += 1
        elif gest[i] == 0 and gref[i] == 0:
            TN += 1
        elif gest[i] == 0 and gref[i] == 1:
            FN += 1

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    return TPR, FPR, Precision


def compute_lam_single_process(ts, Gref, method, dataset):
    grefs, gests = [], []
    method_name = method.split('_')[2]
    method = getattr(sys.modules[__name__], method)
    print(method_name)
    for subj, (_ts, _gref) in tqdm(enumerate(zip(ts, Gref)), total=len(ts),
                                   desc=f'{method.__name__.split("_")[-1].capitalize()}'):
        X = torch.tensor(_ts[np.newaxis], dtype=torch.float32, device=device).cpu().numpy()
        # lams = np.linspace(0.1, 1, 10, endpoint=True)
        # lams = [0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1]
        lams = param_config[dataset][method_name]['lam']
        lams = np.linspace(lams[0], lams[1], 10, endpoint=True)

        Gests = [method((lam, X, _gref, dataset)) for lam in tqdm(lams, desc=f'Subject {subj}')]

        # sort the Gests by lambda
        Gests = sorted(Gests, key=lambda x: x[0])

        N = X.shape[-1]
        summary_graph = np.ones_like(_gref) * np.inf
        for i in range(N):
            for j in range(N):
                if i == j:
                    summary_graph[i, j] = 0
                    _gref[i, i] = 0

                for lam, Gest in Gests:
                    if Gest[i, j] == 0 and lam < summary_graph[i, j]:
                        summary_graph[i, j] = lam

        grefs.append(np.where(_gref != 0, 1, 0))
        gests.append(summary_graph)

    return grefs, gests


def compute_lam_multi_process(ts, Gref, method, dataset):
    grefs, gests = [], []
    method_name = method.split('_')[2]
    method = getattr(sys.modules[__name__], method)
    print(method_name)
    for subj, (_ts, _gref) in tqdm(enumerate(zip(ts, Gref)), total=len(ts), desc=f'{method.__name__.split("_")[-1].capitalize()}'):
        X = torch.tensor(_ts[np.newaxis], dtype=torch.float32, device=device).cpu().numpy()
        # lams = np.linspace(0.1, 1, 10, endpoint=True)
        # lams = [0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1]
        lams = param_config[dataset][method_name]['lam']
        lams = np.linspace(lams[0], lams[1], 10, endpoint=True)

        # set number of proctrainVerboseLvlesses
        with mp.Pool(10) as pool:
            Gests = pool.map(method, [(lam, X, _gref, dataset) for lam in lams])
        # Gests = [method((lam, X, _gref, dataset)) for lam in tqdm(lams, desc=f'Subject {subj}')]

        # sort the Gests by lambda
        Gests = sorted(Gests, key=lambda x: x[0])

        N = X.shape[-1]
        summary_graph = np.ones_like(_gref) * np.inf
        for i in range(N):
            for j in range(N):
                if i == j:
                    summary_graph[i, j] = 0
                    _gref[i, i] = 0

                for lam, Gest in Gests:
                    if Gest[i, j] == 0 and lam < summary_graph[i, j]:
                        summary_graph[i, j] = lam

        grefs.append(np.where(_gref != 0, 1, 0))
        gests.append(summary_graph)

    return grefs, gests


if __name__ == '__main__':
    mp.set_start_method('spawn')

    fileName = f".cache/netsim/sim3.mat"
    test_data = scio.loadmat(fileName)
    ts = [test_data['ts'][i * 200:(i + 1) * 200] for i in range(2, 3)]
    Gref = [test_data['net'][i] for i in range(2, 3)]
    method = "compute_lam_esru_2LF"

    grefs, gests = compute_lam_multi_process(ts, Gref, method=method, dataset="netsim")

    score_auroc, score_auprc = [], []
    for i in range(len(grefs)):
        score_auroc.append(roc_auc_score(grefs[i].flatten(), gests[i].flatten()))
        score_auprc.append(average_precision_score(grefs[i].flatten(), gests[i].flatten()))
    mean_auroc = np.mean(score_auroc)
    mean_auprc = np.mean(score_auprc)

    print(f"AUROC: {mean_auroc} AUPRC: {mean_auprc}")

    # auroc, auprc = compute_lam_by_sweeping(ts, Gref, method=method, dataset="netsim")
    # print(f"AUROC: {auroc} AUPRC: {auprc}")
