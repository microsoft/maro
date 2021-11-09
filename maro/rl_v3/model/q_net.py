from abc import ABCMeta, abstractmethod
from typing import Optional

import torch

from maro.rl_v3.model import AbsNet
from maro.rl_v3.utils import SHAPE_CHECK_FLAG, match_shape


class QNet(AbsNet):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(QNet, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor] = None) -> bool:
        if not SHAPE_CHECK_FLAG:
            return True
        else:
            if states.shape[0] == 0 or not match_shape(states, (None, self.state_dim)):
                return False
            if actions is not None:
                if not match_shape(actions, (states.shape[0], self.action_dim)):
                    return False
            return True

    def q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        assert self._shape_check(states=states, actions=actions)
        q = self._get_q_values(states, actions)
        assert match_shape(q, (states.shape[0], 1))  # [B, 1]
        return q

    @abstractmethod
    def _get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pass


class DiscreteQNet(QNet):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscreteQNet, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    @property
    def action_num(self) -> int:
        return self._action_num

    def q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        assert self._shape_check(states=states)
        q = self._get_q_values_for_all_actions(states)
        assert match_shape(q, (states.shape[0], self.action_num))  # [B, action_num]
        return q

    def _get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q = self.q_values_for_all_actions(states)  # [B, action_num]
        return q.gather(1, actions)  # [B, action_num] + [B, 1] => [B, 1]

    @abstractmethod
    def _get_q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        pass


class ContinuousQNet(QNet, metaclass=ABCMeta):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(ContinuousQNet, self).__init__(state_dim=state_dim, action_dim=action_dim)
