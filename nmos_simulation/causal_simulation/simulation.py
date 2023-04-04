import os
import pickle
import numpy as np
from tqdm import tqdm
from sim2600 import params
from argparse import ArgumentParser
from record_transistor_state import

step_limit = 2000


def resample(marked_data):
    # padding to sequence with fix length in a half-clock by the markers (-1) in recording
    marked_data = np.where(marked_data==255, -1, marked_data)
    # detect the end of , last marker point is the end of the sequence
    marked_data = marked_data[:, :np.where(marked_data[0]!=-1)[0][-1]+2]
    # before padding
    print('raw sample length: ', marked_data.shape[1])
    clocks = []
    snippet_lengths = []
    markers = np.where(marked_data[0]==-1)[0].tolist()
    for idx, marker in enumerate(markers):
        if idx == 0:
            clock = marked_data[:, :markers[idx]]
        else:
            clock = marked_data[:, markers[idx-1]+1:markers[idx]]
        steps = clock.shape[1]
        snippet_lengths.append(steps)
        # if the step number in a half-clock is less than step_limit, padding it with the last state to step_limit
        if steps < step_limit:
            if clock.shape[1] == 0:
                clock = np.concatenate((clock, np.tile(clock.reshape(-1, 1), step_limit-steps)), axis=1)
            else:
                clock = np.concatenate((clock, np.tile(clock[:, -1].reshape(-1, 1), step_limit-steps)), axis=1)
        clocks.append(clock)
    marked_data = np.concatenate(clocks, axis=1) if len(clocks) > 1 else clocks[0]
    print('maximum snippet length: ', max(snippet_lengths))
    return marked_data


def get_where_to_perturb(game, root_dir, iteration_time=256, n_perturb=1):
    print('Calculating where to record perturbations')
    if os.path.exists(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}.npy".format(game, iteration_time))):
        # start to calculate where to perturb
        orig = np.load(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}.npy".format(game, iteration_time)), mmap_mode='r')
        u, i = np.unique(orig, return_index=True, axis=0)
        # padding the sequence with fix length in a half-clock by the markers (-1) in recording
        if not os.path.exists(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}_rec_2e3.npy".format(game, iteration_time))):
            orig = resample(np.load(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}.npy".format(game, iteration_time)), mmap_mode='r'))
            np.save(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}_rec_2e3.npy".format(game, iteration_time)), orig)
        else:
            orig = np.load(os.path.join(root_dir, "{}/HR/Regular_3510_step_{}_rec_2e3.npy".format(game, iteration_time)), mmap_mode='r')

        # multiple times of perturbation / only perturb the first time
        if n_perturb == 1:
            # find the first point the sequence turn from 1 to 0 if the first point is 1, and vice versa
            perturb_timepoint = [np.where(seq == 0)[0] if seq[0] == 1 else np.where(seq == 1)[0] for seq in
                                 tqdm(orig, total=orig.shape[0])]
            # if the first point is 1, then force it to 1, vice versa.
            # perturbation works when the following point is 0, and vice versa.
            perturb_type = ['high' if seq[0] == 1 else 'low' for seq in tqdm(orig, total=orig.shape[0])]
            perturb_timepoint = {idx: (arr[0], type) for idx, (arr, type) in enumerate(zip(perturb_timepoint, perturb_type))
                                 if
                                 len(arr) > 0} #  skip the constant sequence (all HIGH or all LOW)
            unique_perturb = {k: (v[0] // step_limit, v[1]) for k, v in perturb_timepoint.items() if k in i}

            with open(os.path.join(root_dir, "{}/perturb_config.pkl".format(game)), "wb") as f:
                pickle.dump(unique_perturb, f, protocol=2)

            return unique_perturb
        else:
            # find perturbation timepoints with fixed interval
            perturb_timepoints = [np.linspace(0, iteration_time, num=n_perturb, endpoint=False).tolist() for seq in orig]
            unique_perturbs = {}  # {idx: [(timepoint, type), (timepoint, type), ...]}
            for idx, seq in enumerate(orig):
                if idx in i and len(perturb_timepoints[idx]) > 0:
                    unique_perturbs[idx] = []
                    for t in perturb_timepoints[idx]:
                        # check the perturb_timepoints in orig is high or low
                        if seq[int(t*step_limit)] == 1:
                            unique_perturbs[idx].append((int(t), 'low'))
                        else:
                            unique_perturbs[idx].append((int(t), 'high'))

            with open(os.path.join(root_dir, "{}/multiple_{}_perturb_config.pkl".format(game, n_perturb)), "wb") as f:
                pickle.dump(unique_perturbs, f, protocol=2)

            return unique_perturbs

    else:
        raise NotImplementedError, 'Calculating where to record perturbations needs the regular state to be recorded'




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--game", type=str, choices=["Pitfall", "DonkeyKong", "SpaceInvaders", "All"], default="All")
    parser.add_argument("--num_iter", type=int, default=256)
    parser.add_argument("--n_perturb", type=int, default=1)
    parser.add_argument("--action", type=str, choices=["High", "Low", "Regular", "Adaptive"], default="Regular")
    parser.add_argument("--save_dir", type=str, default=".cache/sim_data")

    args = parser.parse_args()
    game = args.game
    n_perturb = args.n_perturb
    iteration_time = args.num_iter
    action = args.action
    print(args)
    root_dir = args.save_dir
    game2rom = {"Pitfall": params.ROMS_PITFALL,
                "DonkeyKong": params.ROMS_DONKEY_KONG,
                "SpaceInvaders": params.ROMS_SPACE_INVADERS,
                }

    if game == "All":
        games = ["DonkeyKong", "Pitfall", "SpaceInvaders"]
    else:
        games = [game]

    for game in games:
        if not os.path.exists(os.path.join(root_dir, "{}/HR/".format(game))):
            os.makedirs(os.path.join(root_dir, "{}/HR/".format(game)))
        path = os.path.join(root_dir, "{}/HR/{}_3510_step_{}.npy".format(game, action, iteration_time))
        if not os.path.exists('/'.join(path.split('/')[:-1])):
            os.makedirs('/'.join(path.split('/')[:-1]))

        print("Simulation start!")
        if action != "Regular":
            # get unique transistor index to reduce the number of simulation
            # get where actually to start recording the perturbed data
            if n_perturb == 1:
                perturb_config_path = os.path.join(root_dir, "{}/perturb_config.pkl".format(game))
            else:
                perturb_config_path = os.path.join(root_dir, "{}/multiple_{}_perturb_config.pkl".format(game, n_perturb))
            if os.path.exists(perturb_config_path):
                with open(os.path.join(perturb_config_path), "rb") as f:
                    perturb_config = pickle.load(f)
            else:
                perturb_config = get_where_to_perturb(game, root_dir, iteration_time=iteration_time, n_perturb=n_perturb)

            if n_perturb == 1:
                for tidx, (halfclk, _action) in perturb_config.items():
                    if action == "High":
                        # do high voltage single element lesion analysis
                        data = single_leision_measure_v2(tidx, halfclk, lesion="High", rom=game2rom[game], iteration=iteration_time)
                    elif action == "Low":
                        # do low voltage single element lesion analysis
                        data = single_leision_measure_v2(tidx, halfclk, lesion="Low", rom=game2rom[game], iteration=iteration_time)
                    elif action == "Adaptive":
                        # do adaptive voltage single element lesion analysis
                        data = single_leision_measure_v2(tidx, halfclk, lesion=_action.title(), rom=game2rom[game], iteration=iteration_time)

                    try:
                        path = os.path.join(root_dir, "{}/HR/{}_3510_step_{}_tidx_{}.npy".format(game, action, iteration_time, tidx))
                        np.save(path, data)
                        print("Simulation end and save file at {}!".format(path))
                    except LookupError as e:
                        print("Simulation end but save file failed!\n{e}")
            else:
                assert action == "Adaptive", "Multiple perturbation only support adaptive perturbation"
                for tidx, perturb_points in perturb_config.items():
                    print(perturb_points)
                    for halfclk, _action in perturb_points:
                        data = single_leision_measure_v2(tidx, halfclk, lesion=_action.title(), rom=game2rom[game], iteration=iteration_time)
                        try:
                            if not os.path.exists(os.path.join(root_dir, "{}/HR/multiple_{}/".format(game, n_perturb))):
                                os.makedirs(os.path.join(root_dir, "{}/HR/multiple_{}/".format(game, n_perturb)))
                            path = os.path.join(root_dir, "{}/HR/multiple_{}/{}_3510_step_{}_tidx_{}_halfclk_{}.npy".format(game, n_perturb, action, iteration_time, tidx, halfclk))
                            np.save(path, data)
                            print("Simulation end and save file at {}!".format(path))
                        except LookupError as e:
                            print("Simulation end but save file failed!\n{e}")
        else:
            # Collect regular simulation data
            data = original_measure_hr(rom=game2rom[game], iteration=iteration_time)
            try:
                np.save(path, data)
                print("Simulation end and save file at {}!".format(path))
            except LookupError as e:
                print("Simulation end but save file failed!\n{e}")


