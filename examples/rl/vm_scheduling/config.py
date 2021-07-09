
config = {
    "model": {
        "network": {
            "actor": {
                "input_dim": STATE_DIM,
                "output_dim": NUM_PMS + 1,  # action could be any PM or postponement, hence the plus 1
                "hidden_dims": [64, 32, 32],
                "activation": "leaky_relu",
                "softmax": True,
                "batch_norm": False,
                "head": True
            },
            "critic": {
                "input_dim": STATE_DIM,
                "output_dim": 1,
                "hidden_dims": [256, 128, 64],
                "activation": "leaky_relu",
                "softmax": False,
                "batch_norm": False,
                "head": True
            }
        },
        "optimization": {
            "actor": {
                "optim_cls": "adam",
                "optim_params": {"lr": 0.0001}
            },
            "critic": {
                "optim_cls": "sgd",
                "optim_params": {"lr": 0.001}
            }
        }
    },
    "algorithm": {
        "reward_discount": 0.9,
        "train_epochs": 100,
        "critic_loss_cls": "mse",
        "critic_loss_coeff": 0.1
    },
    "experience_store": {
        "rollout": {"capacity": 10000, "overwrite_type": "rolling"},
        "update": {"capacity": 50000, "overwrite_type": "rolling"}
    },
    "sampler": {
        "rollout": {"batch_size": -1, "replace": False},
        "update": {"batch_size": 128, "replace": True}
    }
}


config = {
    "model": {
        "network": {
            "actor": {
                "input_dim": STATE_DIM,
                "output_dim": NUM_PMS + 1,  # action could be any PM or postponement, hence the plus 1
                "hidden_dims": [64, 32, 32],
                "activation": "leaky_relu",
                "softmax": True,
                "batch_norm": False,
                "head": True
            },
            "critic": {
                "input_dim": STATE_DIM,
                "output_dim": 1,
                "hidden_dims": [256, 128, 64],
                "activation": "leaky_relu",
                "softmax": False,
                "batch_norm": False,
                "head": True
            }
        },
        "optimization": {
            "actor": {
                "optim_cls": "adam",
                "optim_params": {"lr": 0.0001}
            },
            "critic": {
                "optim_cls": "sgd",
                "optim_params": {"lr": 0.001}
            }
        }
    },
    "algorithm": {
        "reward_discount": 0.9,
        "train_epochs": 100,
        "critic_loss_cls": "mse",
        "critic_loss_coeff": 0.1
    },
    "experience_store": {
        "rollout": {"capacity": 10000, "overwrite_type": "rolling"},
        "update": {"capacity": 50000, "overwrite_type": "rolling"}
    },
    "sampler": {
        "rollout": {"batch_size": -1, "replace": False},
        "update": {"batch_size": 128, "replace": True}
    }
}

