# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl.modeling import DiscreteACNet, DiscreteQNet, FullyConnected
from maro.rl.policy import DQN, ActorCritic

cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
from config import (
    ac_conf, actor_net_conf, actor_optim_conf, algorithm, critic_net_conf, critic_optim_conf, dqn_conf, q_net_conf,
    q_net_optim_conf, state_dim
)


class MyQNet(DiscreteQNet):
    def __init__(self):
        super().__init__()
        self.fc = FullyConnected(**q_net_conf)
        self.optim = q_net_optim_conf[0](self.fc.parameters(), **q_net_optim_conf[1])

    @property
    def input_dim(self):
        return state_dim

    @property
    def num_actions(self):
        return q_net_conf["output_dim"]

    def forward(self, states):
        return self.fc(states)

    def step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def get_gradients(self, loss):
        self.optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad):
        for name, param in self.named_parameters():
            param.grad = grad[name]

        self.optim.step()

    def get_state(self):
        return {"network": self.state_dict(), "optim": self.optim.state_dict()}

    def set_state(self, state):
        self.load_state_dict(state["network"])
        self.optim.load_state_dict(state["optim"])


class MyACNet(DiscreteACNet):
    def __init__(self):
        super().__init__()
        self.actor = FullyConnected(**actor_net_conf)
        self.critic = FullyConnected(**critic_net_conf)
        self.actor_optim = actor_optim_conf[0](self.actor.parameters(), **actor_optim_conf[1])
        self.critic_optim = critic_optim_conf[0](self.critic.parameters(), **critic_optim_conf[1])

    @property
    def input_dim(self):
        return state_dim

    @property
    def num_actions(self):
        return q_net_conf["output_dim"]

    def forward(self, states, actor: bool = True, critic: bool = True):
        return (self.actor(states) if actor else None), (self.critic(states) if critic else None)

    def step(self, loss):
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()

    def get_gradients(self, loss):
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad):
        for name, param in self.named_parameters():
            param.grad = grad[name]

        self.actor_optim.step()
        self.critic_optim.step()

    def get_state(self):
        return {
            "network": self.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict()
        }

    def set_state(self, state):
        self.load_state_dict(state["network"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])


if algorithm == "dqn":
    policy_func_dict = {
        f"dqn.{i}": lambda name: DQN(name, MyQNet(), **dqn_conf) for i in range(4)
    }
else:
    policy_func_dict = {
        f"ac.{i}": lambda name: ActorCritic(name, MyACNet(), **ac_conf) for i in range(4)
    }
