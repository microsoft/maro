import os
from shutil import copy2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Config(object):
    def __init__(self):
        self.num_episodes = 30
        self.algorithm = "sac"  # ddpg, sac
        self.experiment_name = f"{self.algorithm}_test"

        self.seed = 1   # Used only if randomize_random_seed is False
        self.randomize_random_seed = True

        self.use_GPU = False

        # Currently the hyperparameters is for Actor_Critic_Agents only, e.g., ddpg, sac
        self.hyperparameters = {
            "Actor": {
                # For NN module
                "linear_hidden_units": [128, 128],
                "output_activation": 'tanh',
                "hidden_activations": 'tanh',
                "dropout": 0.3,
                "initialiser": "Xavier",
                "batch_norm": False,
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
                # For optimizer
                "learning_rate": 0.0003,
                # For Replay Buffer
                "buffer_size": 1000000,
                # For Soft Update
                "tau": 0.005,
                # For gradient clipping
                "gradient_clipping_norm": 5,
            },

            "batch_size": 256,
            # For SAC only
            "automatically_tune_entropy_hyperparameter": True,
            "entropy_term_weight": 0.1, # Used if not automatically tune entropy hyperparameter
            "add_extra_noise": False,   # To apply OU noise or not
            "do_evaluation_iterations": True,
            "min_steps_before_learning": 0,
            # For OU Noise
            "mu": 0.0,
            "theta": 0.05,
            "sigma": 0.05,
            #
            "discount_rate": 0.99,
            "update_every_n_steps": 100,
            "learning_updates_per_learning_session": 2,
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
