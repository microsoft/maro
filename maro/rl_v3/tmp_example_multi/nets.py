from typing import Dict, List

import torch
from torch.optim import Adam, RMSprop

from maro.rl_v3.model import DiscretePolicyNet, FullyConnected, MultiQNet

from .config import action_shaping_conf, state_dim

actor_net_conf = {
    "input_dim": state_dim,
    "hidden_dims": [256, 128, 64],
    "output_dim": len(action_shaping_conf["action_space"]),
    "activation": torch.nn.Tanh,
    "softmax": True,
    "batch_norm": False,
    "head": True
}
critic_conf = {
    "state_dim": state_dim,
    "action_dims": [1]
}
critic_net_conf = {
    "input_dim": critic_conf["state_dim"] + sum(critic_conf["action_dims"]),
    "hidden_dims": [256, 128, 64],
    "output_dim": 1,
    "activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": True,
    "head": True
}
actor_optim_conf = (Adam, {"lr": 0.001})
critic_optim_conf = (RMSprop, {"lr": 0.001})


# #####################################################################################################################
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

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._actor_optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad: dict) -> None:
        for name, param in self.named_parameters():
            param.grad = grad[name]
        self._actor_optim.step()

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

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._critic_optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad: dict) -> None:
        for name, param in self.named_parameters():
            param.grad = grad[name]
        self._critic_optim.step()

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
