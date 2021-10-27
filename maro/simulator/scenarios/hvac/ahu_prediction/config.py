# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

fcn_config = {
    "input_dim": 3,
    "output_dim": 3,
    "hidden_dims": [128, 128, 128],
    "activation": torch.nn.ReLU,
    "dropout_p": 0.15,
}

training_config = {
    "epoch": 2000,
    "batch_size": 2000,
    "shuffle_data": False,
    "learning_rate": 0.01,
    "early_stopping_patience": 1000,
    "loss": torch.nn.MSELoss(),
    "model_path": "/home/Jinyu/maro/maro/simulator/scenarios/hvac/topologies/building121/checkpoints"
}

data_config = {
    "lower_rescale_bound": -1,
    "upper_rescale_bound": 1,
    "seed": 123,
    "split_random": True,
    "train_fraction": 0.7,
    "validation_fraction": 0.2,
    "dataset_path": "/home/Jinyu/maro/maro/simulator/scenarios/hvac/topologies/building121/datasets",
    "input_filename": "train_data_AHU_MAT.csv",
}
