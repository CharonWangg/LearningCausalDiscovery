import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tools.src.utils import load_config


if __name__ == "__main__":
    root = "/project/learning_causal_discovery/run/official_configs_v2"
    base_config = {"transformer": "/project/learning_causal_discovery/run/official_configs_v2/mos6502_transformer_128.py",
                   "tcn": "/project/learning_causal_discovery/run/official_configs_v2/patch_tcn_128_fold_seed_42.py",
                   "lstm": "/project/learning_causal_discovery/run/official_configs_v2/patch_lstm_128_fold_seed_42.py",}
    models = ["transformer", "tcn", "lstm"]

    for model in models:
        for seed in [42, 7, 12, 1207, 3057]:
            _base_config = load_config(base_config[model])
            _base_config.seed = seed
            _base_config.data.train.split = _base_config.data.train.split.replace('fold_seed_42', f'fold_seed_{seed}')
            _base_config.data.val.split = _base_config.data.val.split.replace('fold_seed_42', f'fold_seed_{seed}')
            _base_config.data.test.split = _base_config.data.test.split.replace('fold_seed_42', f'fold_seed_{seed}')
            _base_config.log.exp_name = _base_config.log.exp_name.replace('fold_seed_42', f'fold_seed_{seed}')
            _base_config.dump(os.path.join(root, f"patch_{model}_128_fold_seed_{seed}.py"))