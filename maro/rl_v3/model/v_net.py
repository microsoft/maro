from abc import abstractmethod

import torch

from maro.rl_v3.model import AbsNet
from maro.rl_v3.utils import SHAPE_CHECK_FLAG, match_shape


class VNet(AbsNet):
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
        assert self._shape_check(states)
        v = self._get_v_values(states)
        assert match_shape(v, (states.shape[0],))  # [B]
        return v

    @abstractmethod
    def _get_v_values(self, states: torch.Tensor) -> torch.Tensor:
        pass
