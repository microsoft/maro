# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np
import torch
import torch.nn as nn

from maro.rl.exploration import EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler
from maro.rl.modeling import DiscreteACNet, DiscreteQNet, FullyConnected, OptimOption
from maro.rl.policy import DQN, ActorCritic

cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
from config import ac_conf, ac_net_conf, dqn_conf, q_net_conf, exploration_conf, state_dim


class QNet(DiscreteQNet):
    def __init__(self, component: nn.Module, optim_option: OptimOption=None, device=None):
        super().__init__(component, optim_option=optim_option, device=device)

    @property
    def input_dim(self):
        return state_dim
    
    def forward(self, states):
        return self.component(states)


def get_dqn(name: str):
    qnet = QNet(FullyConnected(**q_net_conf["network"]), optim_option=OptimOption(**q_net_conf["optimization"]))
    exploration = EpsilonGreedyExploration()
    exploration.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        **exploration_conf
    )
    return DQN(name, qnet, exploration=exploration, **dqn_conf)


def get_ac(name: str):
    class MyACNET(DiscreteACNet):
        @property
        def input_dim(self):
            return state_dim

        def forward(self, states, actor: bool = True, critic: bool = True):
            return (
                self.component["actor"](states) if actor else None,
                self.component["critic"](states) if critic else None
            )

    ac_net = MyACNET(
        component={
            "actor": FullyConnected(**ac_net_conf["network"]["actor"]),
            "critic": FullyConnected(**ac_net_conf["network"]["critic"])
        },
        optim_option={
            "actor":  OptimOption(**ac_net_conf["optimization"]["actor"]),
            "critic": OptimOption(**ac_net_conf["optimization"]["critic"])
        }
    )

    return ActorCritic(name, ac_net, **ac_conf)


policy_func_dict = {
    "dqn": get_dqn,
    "ac.0": get_ac,
    "ac.1": get_ac,
    "ac.2": get_ac
}
