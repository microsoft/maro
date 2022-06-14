# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta
from typing import Tuple

import torch

from maro.rl.model.policy_net import ContinuousPolicyNet


class ContinuousDDPGNet(ContinuousPolicyNet, metaclass=ABCMeta):
    """Policy net for policies that are trained by DDPG.

    The following methods should be implemented:
    - _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:

    Overwrite one or multiple of following methods when necessary.
    - freeze(self) -> None:
    - unfreeze(self) -> None:
    - step(self, loss: torch.Tensor) -> None:
    - get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
    - apply_gradients(self, grad: Dict[str, torch.Tensor]) -> None:
    - get_state(self) -> dict:
    - set_state(self, net_state: dict) -> None:
    """

    def _get_actions_with_probs_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        # Not used in DDPG
        pass

    def _get_actions_with_logps_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        # Not used in DDPG
        pass

    def _get_states_actions_probs_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Not used in DDPG
        pass

    def _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Not used in DDPG
        pass
