from abc import ABCMeta, abstractmethod

import torch

from maro.rl_v3.utils import match_shape, SHAPE_CHECK_FLAG
from .abs_net import AbsNet


class VNet(AbsNet, metaclass=ABCMeta):
    """
    Net for V functions.
    """
    def __init__(self, state_dim: int) -> None:
        super(VNet, self).__init__()
        self._state_dim = state_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def _shape_check(self, states: torch.Tensor) -> bool:
        if not SHAPE_CHECK_FLAG:
            return True
        else:
            return states.shape[0] > 0 and match_shape(states, (None, self.state_dim))

    def v_values(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get V-values according to states.

        Args:
            states (torch.Tensor): States.

        Returns:
            V-values with shape [batch_size]
        """
        assert self._shape_check(states), \
            f"States shape check failed. Expecting: {('BATCH_SIZE', self.state_dim)}, actual: {states.shape}."
        v = self._get_v_values(states)
        assert match_shape(v, (states.shape[0],)), \
            f"V-value shape check failed. Expecting: {(states.shape[0],)}, actual: {v.shape}."  # [B]
        return v

    @abstractmethod
    def _get_v_values(self, states: torch.Tensor) -> torch.Tensor:
        """
        Implementation of `v_values`.
        """
        raise NotImplementedError
