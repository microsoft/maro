# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl.exploration import MultiLinearExplorationScheduler
from maro.rl.modeling import DiscreteACNet, DiscreteQNet, FullyConnected
from maro.rl.policy import DQN, ActorCritic

cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
from config import (
    ac_conf, actor_net_conf, actor_optim_conf, critic_net_conf, critic_optim_conf, dqn_conf, q_net_conf,
    q_net_optim_conf, exploration_conf, state_dim
)


class QNet(DiscreteQNet):
    def __init__(self, device=None):
        super().__init__(device=device)
        self.fc = FullyConnected(**q_net_conf)
        self.optim = q_net_optim_conf[0](self.fc.parameters(), **q_net_optim_conf[1])

    @property
    def input_dim(self):
        return state_dim

    def forward(self, states):
        return self.component(states)

    def step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


def get_dqn(name: str):
    exploration = EpsilonGreedyExploration()
    exploration.register_schedule(
        scheduler_cls=MultiLinearExplorationScheduler,
        param_name="epsilon",
        **exploration_conf
    )
    return DQN(name, QNet(), exploration=exploration, **dqn_conf)


def get_ac(name: str):
    class MyACNet(DiscreteACNet):
        def __init__(self):
            self.actor = FullyConnected(**actor_net_conf)
            self.critic = FullyConnected(**critic_net_conf)
            self.actor_optim = actor_optim_conf[0](self.actor.parameters(), **actor_optim_conf[1])
            self.critic_optim = critic_optim_conf[0](self.critic.parameters, **critic_optim_conf[1])

        @property
        def input_dim(self):
            return state_dim

        def forward(self, states, actor: bool = True, critic: bool = True):
            return (self.actor(states) if actor else None), (self.critic(states) if critic else None)

        def step(self, loss):
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()

    return ActorCritic(name, MyACNet(), **ac_conf)


policy_func_dict = {
    "dqn": get_dqn,
    "ac": get_ac
}
