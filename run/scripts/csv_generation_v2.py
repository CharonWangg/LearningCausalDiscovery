# simplified csv file
import copy
import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from argparse import ArgumentParser

noise_seed = 42
seed = [42, 1, 2, 3, 4]

step_limit = 400
num_iterations = 256


set_range = {
    "train": (256, 512),
    "val": (512, 768),
    "test": (768, 1024),
    "extra": (
              # (512, 768), (768, 1024),
              # (1024, 1280),
              # (1280, 1536),
              # (1536, 1792), (1792, 2048),
              # (2048, 2304), (2304, 2560),
              # (2560, 2816), (2816, 3072),
              # (3072, 3328), (3328, 3584),
              # (3584, 3840), (3840, 4096),
              # (4096, 4352), (4352, 4608),
              # (4608, 4864), (4864, 5120),
              # (5120, 5376), (5376, 5632),
              # (5632, 5888), (5888, 6144),
              # (6144, 6400), (6400, 6656),
              # (6656, 6912), (6912, 7168),
              # (7168, 7424), (7424, 7680),
              )

}

range2set = {v: k for k, v in set_range.items()}


def remove_constant_seqs(data, label):
    """
    create the dataframe of transistor pairs
    """
    # if the data is longer than num_iterations, chop it into windows but only use positive samples
    index = list(label.keys())
    data_dict = {i: data[i] for i in index if data[i].std() != 0}

    # only use positive samples from windows

    # get the corresponding label a pair of transistors
    _label = {}
    for i in data_dict:
        for j in data_dict:
            if i == j:
                continue
            if j in label[i]:
                _label[(i, j)] = 1
            else:
                _label[(i, j)] = 0

    # return the df have the transistor id and the label
    df = pd.DataFrame()
    df["id"] = range(len(_label))
    df["transistor_1"] = [key[0] for key, value in _label.items()]
    df["transistor_2"] = [key[1] for key, value in _label.items()]
    df["label"] = [value for key, value in _label.items()]
    return df


def generate_csv(root, extra=False):
    _split = tuple([int(number) for number in root.split("/")[-1].split("_")[-2:]])
    _game = root.split("/")[-3]
    adjacency_matrix = pickle.load(open(os.path.join(root, "adjacency_matrix.pkl"), "rb"))
    file_path = os.path.join(root, f"Regular_3510_step_{num_iterations}_rec_{step_limit}_window_{_split[0]}_{_split[1]}.npy")
    orig = np.load(file_path, mmap_mode="r")
    # not consider the transistors with all 0 or all 1
    print("total non-constant transistors: {}".format(len(adjacency_matrix)))
    df = remove_constant_seqs(
        orig, adjacency_matrix
    )
    df['file_path'] = file_path
    orig_df = copy.deepcopy(df)
    print(df.label.value_counts())

    for _seed in seed:
        if extra:
            tag = 'extra'
        else:
            tag = range2set[_split]
        if tag == 'train':  # for training set
            # add the external data to the original data
            df_ext = get_external_data(game=_game, seed=_seed)
            df = pd.concat((df, df_ext), axis=0).reset_index(drop=True)
            # down sample the negative samples and oversample the positive samples
            df_pos = df[df.label == 1]
            df_neg = df[df.label == 0]
            print("number of positive samples: ", df_pos.shape, "number of negative samples: ", df_neg.shape)
            df_neg = df_neg.sample(frac=train_down_sample_ratio, random_state=_seed)
            df = pd.concat((df_pos, df_neg), axis=0)

            ros = RandomOverSampler(random_state=_seed)
            df_x, df_y = ros.fit_resample(
                df.drop(columns="label"), df["label"]
            )
            df = pd.concat((df_x, df_y), axis=1)
            print("after balancing:\n", df.label.value_counts())

        elif tag == 'extra':
            pass
        else:
            # down sample the samples but keep the ratio of positive and negative samples to be the same
            df_pos = orig_df[orig_df.label == 1]
            df_neg = orig_df[orig_df.label == 0]
            df_pos = df_pos.sample(frac=test_down_sample_ratio, random_state=_seed)
            df_neg = df_neg.sample(frac=test_down_sample_ratio, random_state=_seed)
            df = pd.concat((df_pos, df_neg), axis=0).reset_index(drop=True)

        if not os.path.exists(os.path.join(root, f"csv/fold_seed_{_seed}")):
            os.makedirs(os.path.join(root, f"csv/fold_seed_{_seed}"))
        if tag == 'train':
            df.to_csv(os.path.join(root, f"csv/fold_seed_{_seed}/", f"{tag}_ds_{train_down_sample_ratio}.csv"), index=False)
        else:
            df.to_csv(os.path.join(root, f"csv/fold_seed_{_seed}/", f"{tag}_ds_{test_down_sample_ratio}.csv"), index=False)

        print(f"csv file saved to {os.path.join(root, f'csv/fold_seed_{_seed}/', f'{tag}_ds_{train_down_sample_ratio}.csv')}")

        if extra:
            return df


def get_external_data(game="DonkeyKong", seed=42):
    # get the external data
    df_ext = []
    if not set_range['extra']:
        return pd.DataFrame()

    if isinstance(set_range['extra'][0], int):
        df_ext = generate_csv(os.path.join(root_dir, f"{game}/HR/window_{set_range['extra'][0]}_{set_range['extra'][1]}"), extra=True, seed=seed)
    elif isinstance(set_range['extra'][0], tuple):
        for start, end in set_range['extra']:
            df = generate_csv(os.path.join(root_dir, f"{game}/HR/window_{start}_{end}"), extra=True, seed=seed)
            df_ext.append(df)
        df_ext = pd.concat(df_ext, axis=0).reset_index(drop=True)
    return df_ext


def get_noised_data(root, tag, _seed=42):
    noise_scale = [0.1, 0.3, 0.5]
    start, end = set_range[tag]
    file_path = os.path.join(root, f"HR/window_{start}_{end}/"
                                  f"Regular_3510_step_{num_iterations}_rec_{step_limit}_window_{start}_{end}.npy")
    orig = np.load(file_path, mmap_mode="r")
    np.random.seed(_seed)
    for scale in noise_scale:
        noised_orig = copy.deepcopy(orig).astype(np.float64)
        for i in range(orig.shape[0]):
            noised_orig[i] += np.random.normal(loc=0, scale=scale, size=noised_orig[i].shape)

        np.save(os.path.join(root, f"HR/window_{start}_{end}/"
                                   f"Regular_3510_step_{num_iterations}_rec_{step_limit}_window_{start}_{end}_Noise_{scale}.npy"), noised_orig)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=".cache/sim_data/")
    parser.add_argument("--train_down_sample_ratio", type=float, default=0.01)
    parser.add_argument("--test_down_sample_ratio", type=float, default=0.01)

    args = parser.parse_args()
    root_dir = args.save_dir
    train_down_sample_ratio = args.train_down_sample_ratio
    test_down_sample_ratio = args.test_down_sample_ratio

    # generate the train/valid/test csv for DonkeyKong
    # train:  256 - 512 clk, val: 512 - 768 clk, test: 768 - 1024 clk
    for split in ["train", "val", "test"]:
        files_dir = os.path.join(root_dir, f"DonkeyKong/HR/window_{set_range[split][0]}_{set_range[split][1]}")
        generate_csv(files_dir)

    # generate the test csv for Pitfall
    # test: 768 - 1024 clk
    files_dir = os.path.join(root_dir, f"Pitfall/HR/window_{set_range['test'][0]}_{set_range['test'][1]}")
    generate_csv(files_dir)

    # generate the test csv for SpaceInvaders
    # test: 768 - 1024 clk
    files_dir = os.path.join(root_dir, f"SpaceInvaders/HR/window_{set_range['test'][0]}_{set_range['test'][1]}")
    generate_csv(files_dir)

    # generate the additive noise data
    get_noised_data(os.path.join(root_dir, "DonkeyKong"), "test", _seed=noise_seed)


