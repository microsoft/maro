import os
from shutil import copy2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Config(object):
    def __init__(self):
        self.num_episodes_to_run = 30
        self.algorithm = "sac"  # ddpg, sac
        self.experiment_name = f"{self.algorithm}_test"

        self.seed = 1   # Used only if randomize_random_seed is False
        self.randomize_random_seed = True

        self.standard_deviation_results = 1.0

        self.use_GPU = False

        # Currently the hyperparameters is for Actor_Critic_Agents only, e.g., ddpg, sac
        self.hyperparameters = {
            "Actor": {
                "learning_rate": 0.0003,
                "linear_hidden_units": [128, 128],
                "hidden_activations": 'tanh',
                "dropout": 0.3,
                "final_layer_activation": 'tanh',
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

            "Critic": {
                "learning_rate": 0.0003,
                "linear_hidden_units": [128, 128],
                "hidden_activations": 'tanh',
                "dropout": 0.3,
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 1000000,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

            "min_steps_before_learning": 5000,
            "batch_size": 256,
            "discount_rate": 0.99,
            "mu": 0.0,      # for O-H noise
            "theta": 0.05,  # for O-H noise
            "sigma": 0.05,  # for O-H noise
            "update_every_n_steps": 100,
            "learning_updates_per_learning_session": 2,
            "automatically_tune_entropy_hyperparameter": True,  # SAC
            "entropy_term_weight": 0.1, # SAC
            "add_extra_noise": False,   # SAC
            "do_evaluation_iterations": True,   # SAC
            "clip_rewards": False
        }

        copy2(src=os.path.abspath(__file__), dst=self.log_dir)

    @property
    def checkpoint_dir(self):
        checkpoint_dir = os.path.join(CURRENT_DIR, "checkpoints", self.experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir

    @property
    def log_dir(self):
        log_dir = os.path.join(CURRENT_DIR, "logs", self.experiment_name)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
