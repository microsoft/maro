from typing import Dict, List

import torch

from maro.rl_v3.model import DiscretePolicyNet, FullyConnected, MultiQNet
from maro.rl_v3.policy import DiscretePolicyGradient
from .config import (
    actor_net_conf, actor_optim_conf, algorithm, critic_conf, critic_net_conf, critic_optim_conf, running_mode
)


class MyActorNet(DiscretePolicyNet):
    def __init__(self) -> None:
        super(MyActorNet, self).__init__(state_dim=actor_net_conf["input_dim"], action_num=actor_net_conf["output_dim"])
        self._actor = FullyConnected(**actor_net_conf)
        self._actor_optim = actor_optim_conf[0](self._actor.parameters(), **actor_optim_conf[1])

    def _get_action_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        return self._actor(states)

    def freeze(self) -> None:
        self.freeze_all_parameters()

    def unfreeze(self) -> None:
        self.unfreeze_all_parameters()

    def step(self, loss: torch.Tensor) -> None:
        self._actor_optim.zero_grad()
        loss.backward()
        self._actor_optim.step()

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._actor_optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def get_net_state(self) -> dict:
        return {
            "network": self.state_dict(),
            "actor_optim": self._actor_optim.state_dict()
        }

    def set_net_state(self, net_state: dict) -> None:
        self.load_state_dict(net_state["network"])
        self._actor_optim.load_state_dict(net_state["actor_optim"])


class MyMultiCriticNet(MultiQNet):
    def __init__(self) -> None:
        super(MyMultiCriticNet, self).__init__(
            state_dim=critic_conf["state_dim"],
            action_dims=critic_conf["action_dims"]
        )
        self._critic = FullyConnected(**critic_net_conf)
        self._critic_optim = critic_optim_conf[0](self._critic.parameters(), **critic_optim_conf[1])

    def _get_q_values(self, states: torch.Tensor, actions: List[torch.Tensor]) -> torch.Tensor:
        return self._critic(torch.cat([states] + actions, dim=1)).squeeze(-1)

    def step(self, loss: torch.Tensor) -> None:
        self._critic_optim.zero_grad()
        loss.backward()
        self._critic_optim.step()

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._critic_optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def get_net_state(self) -> dict:
        return {
            "network": self.state_dict(),
            "critic_optim": self._critic_optim.state_dict()
        }

    def set_net_state(self, net_state: dict) -> None:
        self.load_state_dict(net_state["network"])
        self._critic_optim.load_state_dict(net_state["critic_optim"])

    def freeze(self) -> None:
        self.freeze_all_parameters()

    def unfreeze(self) -> None:
        self.unfreeze_all_parameters()


# ###############################################################################
if algorithm == "maac":
    if running_mode == "centralized":
        policies = {
            f"{algorithm}.{i}": DiscretePolicyGradient(
                name=f"{algorithm}.{i}", policy_net=MyActorNet(), device="cpu") for i in range(4)
        }
        get_policy_func_dict = {
            f"{algorithm}.{i}": lambda name: policies[name] for i in range(4)
        }
    elif running_mode == "decentralized":
        get_policy_func_dict = {
            f"{algorithm}.{i}": lambda name: DiscretePolicyGradient(
                name=name, policy_net=MyActorNet(), device="cpu") for i in range(4)
        }
    else:
        raise ValueError
else:
    raise ValueError
