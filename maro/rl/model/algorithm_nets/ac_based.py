# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta
from typing import Tuple

import torch

from maro.rl.model.policy_net import ContinuousPolicyNet, DiscretePolicyNet


class DiscreteACBasedNet(DiscretePolicyNet, metaclass=ABCMeta):
    """Policy net for policies that are trained by Actor-Critic or PPO algorithm and with discrete actions.

    The following methods should be implemented:
    - _get_action_probs_impl(self, states: torch.Tensor) -> torch.Tensor:

    Overwrite one or multiple of following methods when necessary.
    - freeze(self) -> None:
    - unfreeze(self) -> None:
    - step(self, loss: torch.Tensor) -> None:
    - get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
    - apply_gradients(self, grad: Dict[str, torch.Tensor]) -> None:
    - get_state(self) -> dict:
    - set_state(self, net_state: dict) -> None:
    """


class ContinuousACBasedNet(ContinuousPolicyNet, metaclass=ABCMeta):
    """Policy net for policies that are trained by Actor-Critic or PPO algorithm and with continuous actions.

    The following methods should be implemented:
    - _get_actions_with_logps_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    - _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:

    Overwrite one or multiple of following methods when necessary.
    - freeze(self) -> None:
    - unfreeze(self) -> None:
    - step(self, loss: torch.Tensor) -> None:
    - get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
    - apply_gradients(self, grad: Dict[str, torch.Tensor]) -> None:
    - get_state(self) -> dict:
    - set_state(self, net_state: dict) -> None:
    """

    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        actions, _ = self._get_actions_with_logps_impl(states, exploring)
        return actions

    def _get_actions_with_probs_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        # Not used in Actor-Critic or PPO
        pass

    def _get_states_actions_probs_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Not used in Actor-Critic or PPO
        pass
