# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Config(object):
    _instance = None
    _start_time = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            os.environ['TZ'] = "Asia/Shanghai"
            time.tzset()
            cls._instance = object.__new__(cls, *args, **kw)
            cls._start_time = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
        return cls._instance

    def __init__(self):
        self.algorithm = "sac"
        self.num_episode = 100
        self.evaluate_interval = 10

        self.randomize_seed = True
        self.seed = 1       # Used only if randomize_seed is False

        self.device = "cpu"

        self.env_config = {
            "topology": "building121",
            "durations": 500,
        }

        self.state_config = {
            "attributes": ["kw", "at", "dat", "mat"], # Bonsai: ["kw", "at", "mat"],   # The one for Bonsai
            "normalize": True,
            "state_dim": None,    # Would be updated automatically to be len(attributes)
        }

        self.action_config = {
            "action_dim": 2,
            "lower_bound": [0.6, 45],
            "upper_bound": [1.1, 60],
        }

        self.reward_config = {
            # Bonsai
            "type": "Bonsai",  # Bonsai, V2, V3, V4, V5
            # V2
            "V2_efficiency_factor": 10,
            "V2_das_diff_factor": -2,
            "V2_sps_diff_factor": 0,
            "V2_constraints_factor": -0.5,
            "V2_lower_bound": None, # -2.5
            # V3
            "V3_threshold": -5,
            # V4
            "V4_kw_factor": 1,
            "V4_das_diff_penalty_factor": -0.05,
            "V4_dat_penalty_factor": -0.2,
            # V5
            "V5_kw_factor": 4,
            "V5_dat_penalty_factor": -0.06,
        }

        self.hyperparameters = {   # Would be updated automatically according to algorithm used
        }

        self.replay_memory_config = {
            "capacity": 1000000,
            "random_overwrite": False,
            "batch_size": 256,
        }

        self._sac_config = {
            "Actor": {
                # For NN module
                "linear_hidden_units": [128, 128],
                "output_activation": 'tanh',
                "hidden_activations": 'tanh',
                "dropout": 0.3,
                "initialiser": "Xavier",
                "batch_norm": False,
                "seed": 1,
                # For optimizer
                "learning_rate": 0.0003,
                # For soft Update
                "tau": 0.005,
                # For gradient clipping
                "gradient_clipping_norm": 5,
            },

            "Critic": {
                # For NN module
                "linear_hidden_units": [128, 128],
                "output_activation": None,
                "hidden_activations": 'tanh',
                "dropout": 0.3,
                "initialiser": "Xavier",
                "batch_norm": False,
                "base_seed": 1,
                # For optimizer
                "learning_rate": 0.0003,
                # For Soft Update
                "tau": 0.005,
                # For gradient clipping
                "gradient_clipping_norm": 5,
            },

            "sac": {
                "reward_discount": 0.99,
                "soft_update_coeff": 0.995,     # 1 - tau
                "alpha": 0.1,
                "automatically_tune_entropy_hyperparameter": True,
                "warmup": 0,
                "update_target_every": 1,
                "num_training_epochs": 1,
                "add_ou_noise": False,
            },
        }

        self._ddpg_config = {
            "Actor": {
                # For NN module
                "linear_hidden_units": [128, 128],
                "output_activation": 'tanh',
                "hidden_activations": 'tanh',
                "dropout": 0.3,
                "initialiser": "Xavier",
                "batch_norm": False,
                "seed": 1,
                # For optimizer
                "learning_rate": 0.0003,
                # For soft Update
                "tau": 0.005,
                # For gradient clipping
                "gradient_clipping_norm": 5,
            },

            "Critic": {
                # For NN module
                "linear_hidden_units": [128, 128],
                "output_activation": None,
                "hidden_activations": 'tanh',
                "dropout": 0.3,
                "initialiser": "Xavier",
                "batch_norm": False,
                "base_seed": 1,
                # For optimizer
                "learning_rate": 0.0003,
                # For Soft Update
                "tau": 0.005,
                # For gradient clipping
                "gradient_clipping_norm": 5,
            },

            "ddpg": {
                "reward_discount": 0.99,
                "soft_update_coeff": 0.995,
                "warmup": 0,
                "update_target_every": 1,
                "num_training_epochs": 2,
            },
        }

        self.ou_noise_config = {
            "seed": 1,
            "mu": 0.0,
            "theta": 0.05,
            "sigma": 0.05,
        }

        self.checkpoint_dir = os.path.join(CURRENT_DIR, "checkpoints", self.experiment_name)
        self.log_dir = os.path.join(CURRENT_DIR, "logs", self.experiment_name)
        self.baseline_path = os.path.join(
            CURRENT_DIR,
            "../../../maro/simulator/scenarios/hvac/topologies/building121/datasets/train_data_AHU_MAT.csv"
        )

        self._update_values_accordingly()

    @property
    def experiment_name(self):
        return (
            f"{self._start_time}_"
            f"{self.algorithm}_"
            f"{self.reward_config['type']}_"
            f"{self.num_episode}_"
            f"test"
        )

    def _update_values_accordingly(self):
        # state config
        self.state_config["state_dim"] = len(self.state_config["attributes"])

        # random seed
        if self.randomize_seed:
            self.seed = random.randint(0, 2**32 - 2)
            self._sac_config["Actor"]["seed"] = random.randint(0, 2**32 - 2)
            self._sac_config["Critic"]["base_seed"] = random.randint(0, 2**32 - 2)

        # hyperparameter
        if self.algorithm == "sac":
            self.hyperparameters.update(self._sac_config)
        elif self.algorithm == "ddpg":
            self.hyperparameters.update(self._ddpg_config)
        else:
            print(f"Wrong algorithm name: {self.algorithm}, please choose from [sac, ddpg]")
            exit(0)

        # folder
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


config = Config()

if config.algorithm == "sac":
    from .sac_policy import policy_func_dict
    policy_func_dict = policy_func_dict
elif config.algorithm == "ddpg":
    from .ddpg_policy import policy_func_dict
    policy_func_dict = policy_func_dict
