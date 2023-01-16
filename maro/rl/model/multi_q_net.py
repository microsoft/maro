# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod
from typing import List

import torch

from maro.rl.utils import SHAPE_CHECK_FLAG, match_shape

from .abs_net import AbsNet


class MultiQNet(AbsNet, metaclass=ABCMeta):
    """Abstract net for multi-agent Q functions.

    Args:
        state_dim (int): Dimension of states.
        action_dims (List[int]): Dimensions of Dimension of multi-agents' actions. Its length equals the
            number of agents.
    """

    def __init__(self, state_dim: int, action_dims: List[int]) -> None:
        super(MultiQNet, self).__init__()
        self._state_dim = state_dim
        self._action_dims = action_dims

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dims(self) -> List[int]:
        return self._action_dims

    @property
    def agent_num(self) -> int:
        return len(self._action_dims)

    def _shape_check(self, states: torch.Tensor, actions: List[torch.Tensor] = None) -> bool:
        """Check whether the states and actions have valid shapes.

        Args:
            states (torch.Tensor): State tensor.
            actions (List[torch.Tensor], default=None): Action tensors. It length must be equal to the number of agents.
                If it is None, it means we only check state tensor's shape.

        Returns:
            valid_flag (bool): whether the states and actions have valid shapes.
        """
        if not SHAPE_CHECK_FLAG:
            return True
        else:
            if states.shape[0] == 0 or not match_shape(states, (None, self.state_dim)):
                return False
            if actions is not None:
                if len(actions) != self.agent_num:
                    return False
                for action, dim in zip(actions, self.action_dims):
                    if not match_shape(action, (states.shape[0], dim)):
                        return False
            return True

    def q_values(self, states: torch.Tensor, actions: List[torch.Tensor]) -> torch.Tensor:
        """Get Q-values according to states and actions.

        Args:
            states (torch.Tensor): States.
            actions (List[torch.Tensor]): List of actions.

        Returns:
            q (torch.Tensor): Q-values with shape [batch_size].
        """
        assert self._shape_check(states, actions)
        q = self._get_q_values(states, actions)
        assert match_shape(
            q,
            (states.shape[0],),
        ), f"Q-value shape check failed. Expecting: {(states.shape[0],)}, actual: {q.shape}."  # [B]
        return q

    @abstractmethod
    def _get_q_values(self, states: torch.Tensor, actions: List[torch.Tensor]) -> torch.Tensor:
        """Implementation of `q_values`."""
        raise NotImplementedError
