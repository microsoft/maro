# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.optim import Adam

from maro.rl import OptimOption

from examples.cim.common import ACTION_SPACE

agent_config = {
    "hyper_params": {
        "reward_discount": 0.99,
        "td_steps": 100,
        "actor_loss_coefficient": 0.1,
        "entropy_factor": 0.1
    },
    "model": {
        "device": "cpu",
        "scale": 4,
        "sequence_buffer_size": 20,
        "hidden_size": 32,
        "port_edge_feature_dim": 1, 
        "vessel_edge_feature_dim": 2,
        "port_pre_dim": 16,
        "vessel_pre_dim": 8,
        "policy_hidden_dims": [16],
        "value_hidden_dims": [16],
        "graph_output_dim": 32,
        "action_dim": len(ACTION_SPACE)
    },
    "optimization": OptimOption(optim_cls=Adam, optim_params={"lr": 0.00005}),
    "attention_order": "temporal", 
    "onehot_identity": False
}
