# simplified csv file
import copy
import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from imblearn.over_sampling import RandomOverSampler
from argparse import ArgumentParser

step_limit = 2000


def remove_constant_seqs(data, label, index):
    data_dict = {i: data[i] for i in index if data[i].std() != 0}
    # get the corresponding label a pair of transistors
    _label = {}
    for i in data_dict:
        for j in data_dict:
            if i == j: continue
            _label[(i, j)] = 1 if j in label[i] else 0
    # return the df have the transistor id and the label
    df = pd.DataFrame()
    df["id"] = range(len(_label))
    df["transistor_1"] = [key[0] for key, value in _label.items()]
    df["transistor_2"] = [key[1] for key, value in _label.items()]
    df["label"] = [value for key, value in _label.items()]
    return df


def resample(marker_data):
    marker_data = np.where(marker_data==255, -1, marker_data)
    marker_data = marker_data[:, :np.where(marker_data[0]!=-1)[0][-1]+2]
    print('raw sample length: ', marker_data.shape[1])
    clocks = []
    snippet_lengths = []
    markers = np.where(marker_data[0]==-1)[0].tolist()
    for idx, marker in enumerate(markers):
        if idx == 0:
            clock = marker_data[:, :markers[idx]]
        else:
            clock = marker_data[:, markers[idx-1]+1:markers[idx]]
        steps = clock.shape[1]
        snippet_lengths.append(steps)
        if steps < step_limit:
            if clock.shape[1] == 0:
                clock = np.concatenate((clock, np.tile(clock.reshape(-1, 1), step_limit-steps)), axis=1)
            else:
                clock = np.concatenate((clock, np.tile(clock[:, -1].reshape(-1, 1), step_limit-steps)), axis=1)
        clocks.append(clock)
    marker_data = np.concatenate(clocks, axis=1) if len(clocks) > 1 else clocks[0]
    print('maximum snippet length: ', max(snippet_lengths))
    return marker_data


def get_unique_transistors_across_games(games, interval=10, root_dir='.cache/sim_data/'):
    mats = []
    idxs = {}
    for game in tqdm(games):
        mat = resample(np.load(os.path.join(root_dir, '{}/HR/Regular_3510_step_256.npy'.format(game)),
                      mmap_mode='r'))[:, ::interval]
        # mark the unique pairs
        mat, idx = np.unique(mat, axis=0, return_index=True)
        mats.append(mat)
        idxs[game] = idx
    game_types = sum([[game] * mat.shape[0] for game, mat in zip(games, mats)], [])
    mats = np.concatenate(mats, axis=0)
    mat_idx2game_idx = dict(zip(range(mats.shape[0]),
                                zip(np.concatenate([game_idx for idx, (game, game_idx) in enumerate(idxs.items())]),
                                    game_types)))
    mat, idx = np.unique(mats, return_index=True, axis=0)
    unique_game_idx = [mat_idx2game_idx[i] for i in idx]
    unique_game_transistors = {game: [game_idx for game_idx, game_type in unique_game_idx if game_type == game]
                               for game in games}
    return unique_game_transistors


def generate_csv(game, seed=42, interval=10, split=True, save_dir='.cache/sim_data/'):
    print(f"Processing game {game} now.")
    if isinstance(game, list):
        game_indexes = get_unique_transistors_across_games(game, interval, save_dir)
    else:
        meta_data = resample(np.load(os.path.join(save_dir, "{}/HR/Regular_3510_step_256.npy".format(game)), mmap_mode='r'))[:, ::interval]
        unique, indexes = np.unique(meta_data, return_index=True, axis=0)
        game_indexes = {game: indexes}

    for game, indexes in game_indexes.items():
        root_dir = save_dir + "/{}/".format(game)
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        meta_data = resample(np.load(os.path.join(root_dir, "HR/Regular_3510_step_256.npy"), mmap_mode='r'))[:, ::interval]
        indexes = [i for i in indexes if meta_data[i].std() != 0 and i in indexes]
        print('total unique transistors: {}'.format(len(indexes)))
        if split:
            train_index, test_index = train_test_split(indexes, test_size=0.4, random_state=seed)
            valid_index, test_index = train_test_split(test_index, test_size=0.5, random_state=seed)
            print('Number of unique transistors in train: ', len(train_index))
            print('Number of unique transistors in valid: ', len(valid_index))
        else:
            test_index = copy.deepcopy(indexes)
        print('Number of unique transistors in test: ', len(test_index))

        label = pickle.load(open(os.path.join(root_dir, "adjacency_matrix.pkl"), "rb"))
        if split:
            train_df = remove_constant_seqs(meta_data, label, train_index)  # exclude the constant and approximately constant seq
            valid_df = remove_constant_seqs(meta_data, label, valid_index)
            # balance the causal and non-causal data
            ros = RandomOverSampler(random_state=seed)
            print(train_df.label.value_counts())
            train_x, train_y = ros.fit_resample(train_df.drop(columns='label'), train_df['label'])
            train_df = pd.concat((train_x, train_y), axis=1)
        test_df = remove_constant_seqs(meta_data, label, test_index)

        if not os.path.exists(os.path.join(root_dir, f"csv/fold_seed_{seed}")):
            os.makedirs(os.path.join(root_dir, f"csv/fold_seed_{seed}"))

        if split:
            train_df.to_csv(os.path.join(root_dir, f"csv/fold_seed_{seed}/", "train_sim_grouped.csv"), index=False)
            valid_df.to_csv(os.path.join(root_dir, f"csv/fold_seed_{seed}/", "valid_sim_grouped.csv"), index=False)
            test_df.to_csv(os.path.join(root_dir, f"csv/fold_seed_{seed}/", "test_sim_grouped.csv"), index=False)
        else:
            test_df.to_csv(os.path.join(root_dir, f"csv/fold_seed_{seed}/", "all_sim_grouped.csv"), index=False)

        if split:
            print(f"game {game}: train: {len(train_df)}, valid: {len(valid_df)}, test: {len(test_df)}")
        else:
            print(f"game {game}: test: {len(test_df)}")


def test_common_testset_between_across_game_and_single_game():
    # Donkey Kong
    testset_in_single_game = pd.read_csv('/data2/charon/nmos_inference/data_v7/DonkeyKong/test_sim_grouped.csv')
    testset_screened_by_across_game = pd.read_csv('/home/charon/project/learning_causal_discovery/.cache/sim_data/DonkeyKong/csv/fold_seed_42/test_sim_grouped.csv')
    np.testing.assert_array_equal(testset_in_single_game.to_numpy(), testset_screened_by_across_game.to_numpy())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--game", type=str, choices=["Pitfall", "DonkeyKong", "SpaceInvaders", "All"], default="All")
    parser.add_argument("--save_dir", type=str, default=".cache/sim_data/")

    args = parser.parse_args()
    game = args.game
    root_dir = args.save_dir
    if game == "All":
        games = ["DonkeyKong", "Pitfall", "SpaceInvaders"]
    else:
        games = [game]

    for seed in [42, 7, 12, 1207, 3057]:
        # generate the csv after removing the constant and repeated transistors across games
        generate_csv(games, seed, interval=10, save_dir=root_dir, split=True)