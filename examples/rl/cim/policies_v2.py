# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from typing import Any

import torch

from maro.rl.modeling import FullyConnected
from maro.rl.modeling_v2 import DiscretePolicyGradientNetwork, DiscreteQNetwork, DiscreteQCriticNetwork, DiscreteVActorCriticNet
from maro.rl.policy_v2 import DQN, DiscreteActorCritic, MultiDiscreteActorCritic

cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
from config import (
    ac_conf, actor_net_conf, actor_optim_conf, algorithm, critic_net_conf, critic_optim_conf, dqn_conf, num_policies,
    q_net_conf, q_net_optim_conf, maddpg_conf, centralized_critic_net_conf, centralized_critic_optim_conf, num_agents
)


class MyQNet(DiscreteQNetwork):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self) -> None:
        super(MyQNet, self).__init__(state_dim=q_net_conf["input_dim"], action_num=q_net_conf["output_dim"])
        self.fc = FullyConnected(**q_net_conf)
        self.optim = q_net_optim_conf[0](self.fc.parameters(), **q_net_optim_conf[1])

    def forward(self, x):
        raise NotImplementedError

    def _get_q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        return self.fc(states)

    def step(self, loss: torch.tensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def get_gradients(self, loss: torch.tensor) -> torch.tensor:
        self.optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad: dict) -> None:
        for name, param in self.named_parameters():
            param.grad = grad[name]

        self.optim.step()

    def get_state(self) -> object:
        return {"network": self.state_dict(), "optim": self.optim.state_dict()}

    def set_state(self, state: dict) -> None:
        self.load_state_dict(state["network"])
        self.optim.load_state_dict(state["optim"])


class MyACNet(DiscreteVActorCriticNet):
    def __init__(self) -> None:
        super(MyACNet, self).__init__(state_dim=actor_net_conf["input_dim"], action_num=actor_net_conf["output_dim"])
        self.actor = FullyConnected(**actor_net_conf)
        self.critic = FullyConnected(**critic_net_conf)
        self.actor_optim = actor_optim_conf[0](self.actor.parameters(), **actor_optim_conf[1])
        self.critic_optim = critic_optim_conf[0](self.critic.parameters(), **critic_optim_conf[1])

    def _get_v_critic(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(states).squeeze(-1)

    def _get_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        return self.actor(states)

    def step(self, loss: torch.tensor) -> None:
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()
        # a = []
        # print(a[0])

    def get_gradients(self, loss: torch.tensor) -> torch.tensor:
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad: dict) -> None:
        for name, param in self.named_parameters():
            param.grad = grad[name]

        self.actor_optim.step()
        self.critic_optim.step()

    def get_state(self) -> dict:
        return {
            "network": self.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict()
        }

    def set_state(self, state: dict) -> None:
        self.load_state_dict(state["network"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class MyCentralizedCriticNet(DiscreteQCriticNetwork):
    def __init__(self) -> None:
        super(MyCentralizedCriticNet, self).__init__(state_dim=centralized_critic_net_conf["input_dim"], action_num=centralized_critic_net_conf["output_dim"])
        self.fc = FullyConnected(**centralized_critic_net_conf)
        self.optim = centralized_critic_optim_conf[0](self.fc.parameters(), **centralized_critic_optim_conf[1])

    def _get_q_critic_for_all_actions(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([state, action], dim=1))

    def get_gradients(self, loss: torch.tensor) -> torch.tensor:
        self.optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad: dict) -> None:
        for name, param in self.named_parameters():
            param.grad = grad[name]

        self.optim.step()

    def get_state(self) -> dict:
        return {
            "network": self.fc.state_dict(),
            "optim": self.optim.state_dict(),
        }

    def set_state(self, state: dict) -> None:
        self.fc.load_state_dict(state["network"])
        self.optim.load_state_dict(state["optim"])


class MyActorNet(DiscretePolicyGradientNetwork):
    def __init__(self):
        super(MyActorNet, self).__init__(state_dim=actor_net_conf["input_dim"], action_num=actor_net_conf["output_dim"])
        self.fc = FullyConnected(**actor_net_conf)
        self.optim = actor_optim_conf[0](self.fc.parameters(), **actor_optim_conf[1])

    def _get_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        return self.fc(states)

    def get_gradients(self, loss: torch.tensor) -> torch.tensor:
        self.optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad: dict) -> None:
        for name, param in self.named_parameters():
            param.grad = grad[name]

        self.optim.step()

    def get_state(self) -> dict:
        return {
            "network": self.fc.state_dict(),
            "optim": self.optim.state_dict(),
        }

    def set_state(self, state: dict) -> None:
        self.fc.load_state_dict(state["network"])
        self.optim.load_state_dict(state["optim"])


if algorithm == "dqn":
    policy_func_dict = {
        f"{algorithm}.{i}": lambda name: DQN(name, MyQNet(), **dqn_conf) for i in range(num_policies)
    }
elif algorithm == "ac":
    policy_func_dict = {
        f"{algorithm}.{i}": lambda name: DiscreteActorCritic(name, MyACNet(), **ac_conf) for i in range(num_policies)
    }
elif algorithm == "maddpg":
    agent_nets = [MyActorNet() for i in range(num_agents)]
    maddpg_policy = MultiDiscreteActorCritic(algorithm, agent_nets, MyCentralizedCriticNet(), **maddpg_conf)
    policy_func_dict = {f"{algorithm}": lambda name: maddpg_policy}
else:
    raise ValueError
