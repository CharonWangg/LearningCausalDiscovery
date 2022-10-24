import os
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
from argparse import ArgumentParser
from simulation import resample

n_windows = 5
n_iters = 512
# interpolation for each clock
step_limit = 2000
interval = 10


def get_config(roo_dir='.cache/sim_data'):
    # get the config recorded how and when to perturb which transistor
    with open(os.path.join(roo_dir, "{}/perturb_config.pkl".format(game)), "rb") as f:
        config = pickle.load(f)
    return config


def get_orig(game, root_dir):
    if os.path.exists(os.path.join(root_dir, "{}/HR/Regular_3510_step_256_rec_2e3.npy".format(game))):
        orig = np.load(os.path.join(root_dir, "{}/HR/Regular_3510_step_256_rec_2e3.npy".format(game)), mmap_mode='r')
    else:
        orig = np.load(os.path.join(root_dir, "{}/HR/Regular_3510_step_256.npy".format(game)), mmap_mode='r')
        orig = resample(orig)
        np.save(os.path.join(root_dir, "{}/HR/Regular_3510_step_256_rec_2e3.npy".format(game)), orig)
    return orig


def get_label(game, root_dir):
    unique_perturb = get_config()
    potential_resultant = defaultdict(list)
    orig = get_orig(game, root_dir)[:, ::interval]
    for idx in tqdm(unique_perturb):
        # load the perturbed data for transistor idx
        perturb = np.load(os.path.join(root_dir, "{}/HR/Adaptive_3510_step_256_tidx_{}.npy".format(game, idx)), mmap_mode='r')
        # padding to the fix length of one half-clock, last point is marker (-1)
        padded_perturb = np.concatenate(
            (perturb[:, :-1], np.tile(perturb[:, -2].reshape(-1, 1), step_limit - perturb.shape[1] + 1)), axis=1)[:, ::interval]
        # compare the regular state and 'perturbed' state when the cause transistor (idx) is perturbed
        for i in range(orig.shape[0]):
            # skip the cause transistor
            if i != idx:
                # detect the first point where the perturbation actually works
                if unique_perturb[idx][1] == 'high':
                    div_point = np.where(orig[idx] != 1)[0][0]
                else:
                    div_point = np.where(orig[idx] != 0)[0][0]
                # calculate the difference between the regular and the 'perturbed' in the half-clock perturbation actually works
                if not (padded_perturb[i][(div_point - (unique_perturb[idx][0] * step_limit//interval)):] == orig[i, div_point: ((unique_perturb[idx][0] + 1) * step_limit//interval)]).all() and \
                        (padded_perturb[i][:(div_point - (unique_perturb[idx][0] * step_limit//interval))] == orig[i, (unique_perturb[idx][0] * step_limit//interval):div_point]).all():
                    potential_resultant[idx].append(i)
    return potential_resultant


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--game", type=str, choices=["Pitfall", "DonkeyKong", "SpaceInvaders", "All"], default="All")
    parser.add_argument("--save_dir", type=str, default=".cache/sim_data")

    args = parser.parse_args()
    game = args.game
    root_dir = args.save_dir

    if game == "All":
        games = ["DonkeyKong", "Pitfall", "SpaceInvaders"]
    else:
        games = [game]

    for game in games:
        label = get_label(game, root_dir)
        with open(os.path.join(root_dir, "{}/adjacency_matrix.pkl".format(game)), "wb") as f:
            pickle.dump(label, f)

