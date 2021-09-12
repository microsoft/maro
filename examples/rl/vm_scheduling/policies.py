
import sys
from os.path import dirname, realpath

import torch

from maro.rl.modeling import DiscreteACNet, DiscreteQNet, FullyConnected
from maro.rl.policy import DQN, ActorCritic

vm_path = dirname(realpath(__file__))
sys.path.insert(0, vm_path)
from config import (
    ac_conf, actor_net_conf, actor_optim_conf, algorithm, critic_net_conf, critic_optim_conf, dqn_conf, q_net_conf,
    num_features, num_pms, q_net_optim_conf
)


class MyQNet(DiscreteQNet):
    def __init__(self):
        super().__init__()
        for mdl in self.modules():
            if isinstance(mdl, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(mdl.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.fc = FullyConnected(**q_net_conf)
        self.optim = q_net_optim_conf[0](self.fc.parameters(), **q_net_optim_conf[1])

    @property
    def input_dim(self):
        return num_features + num_pms + 1

    @property
    def num_actions(self):
        return q_net_conf["output_dim"]

    def forward(self, states): 
        masks = states[:, num_features:]
        q_for_all_actions = self.fc(states[:, :num_features])
        return q_for_all_actions + (masks - 1) * 1e8

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


class MyACNet(DiscreteACNet):
    def __init__(self):
        super().__init__()
        self.actor = FullyConnected(**actor_net_conf)
        self.critic = FullyConnected(**critic_net_conf)
        self.actor_optim = actor_optim_conf[0](self.actor.parameters(), **actor_optim_conf[1])
        self.critic_optim = critic_optim_conf[0](self.critic.parameters(), **critic_optim_conf[1])

    @property
    def input_dim(self):
        return num_features + num_pms + 1

    def forward(self, states, actor: bool = True, critic: bool = True):
        features = states[:, :num_features].to()
        masks = states[:, num_features:]
        return (self.actor(features) * masks if actor else None), (self.critic(features) if critic else None)

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

if algorithm == "dqn":
    policy_func_dict = {"dqn": lambda name: DQN(name, MyQNet(), **dqn_conf)}
else:
    policy_func_dict = {"ac": lambda name: ActorCritic(name, MyACNet(), **ac_conf)}
