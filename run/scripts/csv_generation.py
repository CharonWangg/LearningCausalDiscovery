# simplified csv file
import copy
import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from argparse import ArgumentParser

interval = 10
step_limit = 400
num_iterations = 256
num_windows = 8


def remove_constant_seqs(data, label, index, train=False):
    """
    create the dataframe of transistor pairs
    """
    # if the data is longer than num_iterations, chop it into windows but only use positive samples
    data_dict = {i: data[i] for i in index if data[i].std() != 0}

    # only use positive samples from windows

    # get the corresponding label a pair of transistors
    _label = {}
    for i in data_dict:
        for j in data_dict:
            if i == j:
                continue
            if j in label[i]:
                # constructing more positive samples for training
                if train:
                    for k in range(num_windows):
                        _label[(i, j, k)] = 1
                else:
                    _label[(i, j, 0)] = 1
            else:
                _label[(i, j, 0)] = 0


    # return the df have the transistor id and the label
    df = pd.DataFrame()
    df["id"] = range(len(_label))
    df["transistor_1"] = [key[0] for key, value in _label.items()]
    df["transistor_2"] = [key[1] for key, value in _label.items()]
    df["window"] = [key[2] for key, value in _label.items()]
    df["label"] = [value for key, value in _label.items()]
    return df


def resample(marker_data):
    marker_data = np.where(marker_data == 255, -1, marker_data)
    marker_data = marker_data[:, : np.where(marker_data[0] != -1)[0][-1] + 2]
    print("raw sample length: ", marker_data.shape[1])
    clocks = []
    snippet_lengths = []
    markers = np.where(marker_data[0] == -1)[0].tolist()
    for idx, marker in enumerate(markers):
        if idx == 0:
            clock = marker_data[:, : markers[idx]]
        else:
            clock = marker_data[:, markers[idx - 1] + 1 : markers[idx]]
        steps = clock.shape[1]
        snippet_lengths.append(steps)
        if steps < step_limit:
            if clock.shape[1] == 0:
                clock = np.concatenate(
                    (clock, np.tile(clock.reshape(-1, 1), step_limit - steps)), axis=1
                )
            else:
                clock = np.concatenate(
                    (clock, np.tile(clock[:, -1].reshape(-1, 1), step_limit - steps)),
                    axis=1,
                )
        clocks.append(clock)
    marker_data = np.concatenate(clocks, axis=1) if len(clocks) > 1 else clocks[0]
    print("maximum snippet length: ", max(snippet_lengths))
    return marker_data


def get_unique_transistors_across_games(
    games, root_dir=".cache/sim_data/"
):
    mats = []
    idxs = {}
    for game in tqdm(games):
        mat = np.load(
                os.path.join(root_dir, f"{game}/HR/Regular_3510_step_{num_iterations}_rec_{step_limit}.npy"),
                mmap_mode="r",
            )[:, ::interval]
        # mark the unique pairs
        mat, idx = np.unique(mat, axis=0, return_index=True)
        mats.append(mat)
        idxs[game] = idx
    game_types = sum([[game] * mat.shape[0] for game, mat in zip(games, mats)], [])
    mats = np.concatenate(mats, axis=0)
    mat_idx2game_idx = dict(
        zip(
            range(mats.shape[0]),
            zip(
                np.concatenate(
                    [game_idx for idx, (game, game_idx) in enumerate(idxs.items())]
                ),
                game_types,
            ),
        )
    )
    mat, idx = np.unique(mats, return_index=True, axis=0)
    unique_game_idx = [mat_idx2game_idx[i] for i in idx]
    unique_game_transistors = {
        game: [game_idx for game_idx, game_type in unique_game_idx if game_type == game]
        for game in games
    }
    return unique_game_transistors


def generate_csv(game, seed=42, split=True, save_dir=".cache/sim_data/"):
    print(f"Processing game {game} now.")
    if isinstance(game, list):
        game_indexes = get_unique_transistors_across_games(game, save_dir)
    else:
        meta_data = np.load(
                os.path.join(save_dir, f"{game}/HR/Regular_3510_step_{num_iterations}_rec_{step_limit}.npy"),
                mmap_mode="r",
            )[:, ::interval]
        print('shape after downsampling: ', meta_data.shape)
        unique, indexes = np.unique(meta_data, return_index=True, axis=0)
        game_indexes = {game: indexes}

    for game, indexes in game_indexes.items():
        root_dir = save_dir + "/{}/".format(game)
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        meta_data = np.load(
                os.path.join(root_dir, f"HR/Regular_3510_step_{num_iterations}_rec_{step_limit}.npy"), mmap_mode="r"
            )[:, ::interval]
        # not consider the transistors with all 0 or all 1
        # indexes = [i for i in indexes if meta_data[i].std() != 0 and i in indexes]
        label = pickle.load(open(os.path.join(root_dir, "adjacency_matrix.pkl"), "rb"))
        indexes = list(label.keys())
        print("total unique transistors: {}".format(len(indexes)))
        if split:
            train_index, test_index = train_test_split(
                indexes, test_size=0.5, random_state=seed
            )
            valid_index, test_index = train_test_split(
                test_index, test_size=0.01, random_state=seed
            )
            print("Number of unique transistors in train: ", len(train_index))
            print("Number of unique transistors in valid: ", len(valid_index))
        else:
            test_index = copy.deepcopy(indexes)
        print("Number of unique transistors in test: ", len(test_index))

        label = pickle.load(open(os.path.join(root_dir, "adjacency_matrix.pkl"), "rb"))
        if split:
            train_df = remove_constant_seqs(
                meta_data, label, train_index, train=True
            )  # exclude the constant and approximately constant seq
            valid_df = remove_constant_seqs(meta_data, label, valid_index)
            # balance the causal and non-causal data
            ros = RandomOverSampler(random_state=seed)
            print(train_df.label.value_counts())
            train_x, train_y = ros.fit_resample(
                train_df.drop(columns="label"), train_df["label"]
            )
            train_df = pd.concat((train_x, train_y), axis=1)
        test_df = remove_constant_seqs(meta_data, label, test_index)

        if not os.path.exists(os.path.join(root_dir, f"csv/fold_seed_{seed}")):
            os.makedirs(os.path.join(root_dir, f"csv/fold_seed_{seed}"))

        if split:
            train_df.to_csv(
                os.path.join(
                    root_dir, f"csv/fold_seed_{seed}/", f"train_sim_grouped_interval_{interval}.csv"
                ),
                index=False,
            )
            valid_df.to_csv(
                os.path.join(
                    root_dir, f"csv/fold_seed_{seed}/", f"valid_sim_grouped_interval_{interval}.csv"
                ),
                index=False,
            )
            test_df.to_csv(
                os.path.join(
                    root_dir, f"csv/fold_seed_{seed}/", f"test_sim_grouped_interval_{interval}.csv"
                ),
                index=False,
            )
        else:
            test_df.to_csv(
                os.path.join(root_dir, f"csv/fold_seed_{seed}/", f"all_sim_grouped_interval_{interval}.csv"),
                index=False,
            )

        if split:
            print(
                f"game {game}: train: {len(train_df)}, valid: {len(valid_df)}, test: {len(test_df)}"
            )
        else:
            print(f"game {game}: test: {len(test_df)}")


def add_noise_to_npy(root_dir, seed, noise_level=[0.1, 0.3, 0.5], game="DonkeyKong"):
    np.random.seed(seed)
    data = np.load(os.path.join(root_dir, f"DonkeyKong/HR/Regular_3510_step_{num_iterations}_rec_{step_limit}.npy"), mmap_mode="r")
    for std in noise_level:
        noised_data = data.copy().astype(np.float32)
        for i in range(data.shape[0]):
            noise = np.random.normal(0, std, data.shape[1])
            noised_data[i] += noise
        np.save(os.path.join(root_dir, f"{game}/HR/Regular_3510_step_{num_iterations}_rec_{step_limit}_noise_{std}_seed_{seed}.npy"), noised_data)
    print(f"noise added on {game} with std {noise_level}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--game",
        type=str,
        choices=["Pitfall", "DonkeyKong", "SpaceInvaders", "All"],
        default="All",
    )
    parser.add_argument("--save_dir", type=str, default=".cache/sim_data/")
    parser.add_argument("--keep_whole", type=bool, default=False)

    args = parser.parse_args()
    game = args.game
    root_dir = args.save_dir
    if args.keep_whole:
        split = False
    else:
        split = True
    if game == "All":
        games = ["DonkeyKong", "Pitfall", "SpaceInvaders"]
    else:
        games = [game]

    for seed in [42, 7, 12, 1207, 3057]:
        # generate the csv after removing the constant and repeated transistors across games
        generate_csv(games, seed, save_dir=root_dir, split=split)
        # generate noised data
        # add_noise_to_npy(root_dir, seed, noise_level=[0.1, 0.3, 0.5], game="DonkeyKong")



