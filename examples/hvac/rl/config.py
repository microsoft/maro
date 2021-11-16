# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Config(object):
    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self):
        self.algorithm = "sac"

        self.randomize_seed = False
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

        self.training_config = {
            # Test
            "test": False,
            # "model_path": "/home/Jinyu/maro/examples/hvac/rl/checkpoints/2021-11-03 04:33:20 ddpg_rewrite_Bonsai_env_positive/ddpg_49",
            "model_path": "/home/Jinyu/maro/examples/hvac/rl/checkpoints/2021-11-03 04:32:29 ddpg_rewrite_V2_env_positive/ddpg_49",
            # Train
            "load_model": False,
            "num_episodes": 30,
            "evaluate_interval": 10,
            "checkpoint_path": os.path.join(CURRENT_DIR, "checkpoints"),
            "log_path": os.path.join(CURRENT_DIR, "logs"),
        }

        self.baseline_path = os.path.join(
            CURRENT_DIR,
            "../../../maro/simulator/scenarios/hvac/topologies/building121/datasets/train_data_AHU_MAT.csv"
        )

        self.hyperparameters = {   # Would be updated automatically according to algorithm used
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

            "replay_memory": {
                "capacity": 1000000,
                "random_overwrite": False,
                "batch_size": 256,
            },

            "sac": {
                "reward_discount": 0.99,
                "soft_update_coeff": 0.995,     # 1 - tau
                "alpha": 0.1,
                "automatically_tune_entropy_hyperparameter": True,
                "warmup": 0,
                "update_target_every": 1,
                "learning_updates_per_learning_session": 2,
                "add_ou_noise": False,
            },

        }

        self.ou_noise_config = {
            "seed": 1,
            "mu": 0.0,
            "theta": 0.05,
            "sigma": 0.05,
        }

        self._update_values_accordingly()

    @property
    def experiment_name(self):
        return f"{self.algorithm}_test"

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

config = Config()
