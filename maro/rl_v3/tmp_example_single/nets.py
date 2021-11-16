from typing import Dict

import torch

from maro.rl_v3.model import DiscretePolicyNet, DiscreteQNet, FullyConnected, VNet
from .config import (
    actor_net_conf, actor_optim_conf, critic_net_conf, critic_optim_conf, q_net_conf, q_net_optim_conf
)


class MyQNet(DiscreteQNet):
    def __init__(self) -> None:
        super(MyQNet, self).__init__(state_dim=q_net_conf["input_dim"], action_num=q_net_conf["output_dim"])
        self._fc = FullyConnected(**q_net_conf)
        self._optim = q_net_optim_conf[0](self._fc.parameters(), **q_net_optim_conf[1])

    def _get_q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        return self._fc(states)

    def step(self, loss: torch.Tensor) -> None:
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def get_net_state(self) -> object:
        return {"network": self.state_dict(), "optim": self._optim.state_dict()}

    def set_net_state(self, net_state: object) -> None:
        assert isinstance(net_state, dict)
        self.load_state_dict(net_state["network"])
        self._optim.load_state_dict(net_state["optim"])

    def freeze(self) -> None:
        self.freeze_all_parameters()

    def unfreeze(self) -> None:
        self.unfreeze_all_parameters()


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


class MyCriticNet(VNet):
    def __init__(self) -> None:
        super(MyCriticNet, self).__init__(state_dim=critic_net_conf["input_dim"])
        self._critic = FullyConnected(**critic_net_conf)
        self._critic_optim = critic_optim_conf[0](self._critic.parameters(), **critic_optim_conf[1])

    def _get_v_values(self, states: torch.Tensor) -> torch.Tensor:
        return self._critic(states).squeeze(-1)

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
