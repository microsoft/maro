# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import datetime
from torch.optim import Adam

from maro.rl import OptimOption

from examples.cim.common import ACTION_SPACE

agent_config = {
    "hyper_params": {
        "reward_discount": 0.99,
        "num_batches": 1,
        "batch_size": 256,
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

training_config = {
    "seed": 1024,
    "env": {
        "scenario": "cim",
        "topology": "global_trade.22p_l0.8",
        "durations": 1120,
    },
    "max_episode": 500,
    "train_freq": 1,
    "model_save_freq": 1,
    "exploration": {
        "parameter_names": ["epsilon"],
        "split": 0.5,
        "start": 0.4,
        "mid": 0.32,
        "end": 0.0
    },
    "group": f"cim-gnn.{datetime.now().timestamp()}",
    "redis_address": ("localhost", "6379"),
    "learner": {
        "update_trigger": 2,
        "inference_trigger": 2
    },
    "actor": {
        "num": 2,
        "receive_action_timeout": 300,  # in milliseconds
        "max_receive_action_attempts": 1,
        "max_null_actions_per_rollout": 15
    }
}