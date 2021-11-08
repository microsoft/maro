# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import torch

topology_name = "building121"

fcn_config = {
    "input_dim": 3,
    "output_dim": 3,
    "hidden_dims": [128, 128, 128],
    "activation": torch.nn.ReLU,
    "dropout_p": 0.15,
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

training_config = {
    "epoch": 2000,
    "batch_size": 2000,
    "shuffle_data": False,
    "learning_rate": 0.01,
    "early_stopping_patience": 1000,
    "loss": torch.nn.MSELoss(),
    "checkpoint_dir": os.path.join(CURRENT_DIR, "checkpoints", topology_name),
}

data_config = {
    "lower_rescale_bound": -1,
    "upper_rescale_bound": 1,
    "seed": 123,
    "split_random": True,
    "train_fraction": 0.7,
    "validation_fraction": 0.2,
    "dataset_dir": os.path.join(CURRENT_DIR, "../topologies", topology_name, "datasets"),
    "input_filename": "train_data_AHU_MAT.csv",
}

test_config = {
    # For test func
    "model_path": os.path.join(training_config["checkpoint_dir"], "best_model.pt"),
    "x_scaler_path": os.path.join(data_config["dataset_dir"], "scaler/x_scaler.joblib"),
    "y_scaler_path": os.path.join(data_config["dataset_dir"], "scaler/y_scaler.joblib"),
    # For baseline
    "is_baseline": True,
    "filepath": os.path.join(data_config["dataset_dir"], "train_data_AHU_MAT.csv"),
    "key_map": {
        "sps": "SPS",
        "das": "DAS",
        "mat_das_delta": "delta_MAT_DAS",
        "kw": "KW",
        "at": "air_ton",
        "dat": "DAT",
    },
    # Images
    "image_dir": training_config["checkpoint_dir"],
    "image_prefix": "baseline_comparison",
}

# test_config = {
#     # For test func
#     "model_path": os.path.join(training_config["checkpoint_dir"], "best_model.pt"),
#     "x_scaler_path": os.path.join(data_config["dataset_dir"], "scaler/x_scaler.joblib"),
#     "y_scaler_path": os.path.join(data_config["dataset_dir"], "scaler/y_scaler.joblib"),
#     # For rollout data
#     "is_baseline": True,
#     "filepath": "/home/Jinyu/maro/examples/hvac/ddpg/logs/2021-11-02 22:41:29 ddpg_V3/data_Train_0.csv",
#     "key_map": {key: key for key in ["sps", "das", "mat", "kw", "at", "dat"]},
#     # Images
#     "image_dir": training_config["checkpoint_dir"],
#     "image_prefix": "rollout_data_ep0",
# }
