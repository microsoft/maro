import os
from shutil import copy2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

actor_critic_agent_hyperparameters = {
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
    "mu": 0.0, #for O-H noise
    "theta": 0.05, #for O-H noise
    "sigma": 0.05, #for O-H noise
    "action_noise_std": 0.1,  # for TD3
    "action_noise_clipping_range": 0.1,  # for TD3
    "update_every_n_steps": 100,
    "learning_updates_per_learning_session": 2,
    "automatically_tune_entropy_hyperparameter": True,
    "entropy_term_weight": 0.1,
    "add_extra_noise": False,
    "do_evaluation_iterations": True,
    "clip_rewards": False
}

dqn_agent_hyperparameters =   {
    "learning_rate": 0.005,
    "batch_size": 128,
    "buffer_size": 40000,
    "epsilon": 1.0,
    "epsilon_decay_rate_denominator": 3,
    "discount_rate": 0.99,
    "tau": 0.01,
    "alpha_prioritised_replay": 0.6,
    "beta_prioritised_replay": 0.1,
    "incremental_td_error": 1e-8,
    "update_every_n_steps": 3,
    "linear_hidden_units": [30, 15],
    "final_layer_activation": "None",
    "batch_norm": False,
    "gradient_clipping_norm": 5,
    "clip_rewards": False,
    "timesteps_to_give_up_control_for": 5
}

class Config(object):
    def __init__(self):
        self.num_episodes_to_run = 30
        self.algorithm = "ddpg"  # ddpg, sac
        self.experiment_name = f"{self.algorithm}_test"

        self.hyperparameters = {
            "Policy_Gradient_Agents": {
                "learning_rate": 0.05,
                "linear_hidden_units": [30, 15],
                "final_layer_activation": "TANH",
                "learning_iterations_per_round": 10,
                "discount_rate": 0.9,
                "batch_norm": False,
                "clip_epsilon": 0.2,
                "episodes_per_learning_round": 10,
                "normalise_rewards": True,
                "gradient_clipping_norm": 5,
                "mu": 0.0,
                "theta": 0.15,
                "sigma": 0.2,
                "epsilon_decay_rate_denominator": 1,
                "clip_rewards": False
            },

            "Actor_Critic_Agents": actor_critic_agent_hyperparameters,
            "DIAYN": {
                "DISCRIMINATOR": {
                    "learning_rate": 0.001,
                    "linear_hidden_units": [32, 32],
                    "final_layer_activation": None,
                    "gradient_clipping_norm": 5
                },
                "AGENT": actor_critic_agent_hyperparameters,
                "MANAGER": dqn_agent_hyperparameters,
                "num_skills": 10,
                "num_unsupservised_episodes": 500
            }
        }

        # Parameters that could kept the stable
        self.seed = 1
        self.randomise_random_seed = True

        self.standard_deviation_results = 1.0

        self.use_GPU = False
        self.runs_per_agent = 1

        self.debug_mode = False
        self.show_solution_score = False
        self.visualise_overall_agent_results = True
        self.visualise_individual_results = False

        self.overwrite_existing_results_file = True
        self.save_model = True

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

    @property
    def file_to_save_data_results(self):
        return os.path.join(self.checkpoint_dir, "results_data.pkl")

    @property
    def file_to_save_model(self, ep=None):
        return os.path.join(self.checkpoint_dir, "model.pk")

    @property
    def file_to_save_results_graph(self):
        return os.path.join(self.log_dir, "results_graph.png")
