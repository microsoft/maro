# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta
from typing import Tuple

import torch

from maro.rl.model.policy_net import ContinuousPolicyNet


class ContinuousSACNet(ContinuousPolicyNet, metaclass=ABCMeta):
    """Policy net for policies that are trained by SAC.

    The following methods should be implemented:
    - _get_actions_with_logps_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:

    Overwrite one or multiple of following methods when necessary.
    - freeze(self) -> None:
    - unfreeze(self) -> None:
    - step(self, loss: torch.Tensor) -> None:
    - get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
    - apply_gradients(self, grad: Dict[str, torch.Tensor]) -> None:
    - get_state(self) -> dict:
    - set_state(self, net_state: dict) -> None:
    """

    def _get_actions_impl(self, states: torch.Tensor, exploring: bool, **kwargs) -> torch.Tensor:
        actions, _ = self._get_actions_with_logps_impl(states, exploring)
        return actions

    def _get_actions_with_probs_impl(
        self,
        states: torch.Tensor,
        exploring: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Not used in SAC
        pass

    def _get_states_actions_probs_impl(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        # Not used in SAC
        pass

    def _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        # Not used in SAC
        pass
