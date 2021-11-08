from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

import torch.nn

from maro.rl_v3.model.abs_net import AbsNet
from maro.rl_v3.utils import SHAPE_CHECK_FLAG, match_shape


class PolicyNet(AbsNet):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(PolicyNet, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def get_actions_with_logps(
        self, states: torch.Tensor, exploring: bool, require_logps: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert self._shape_check(states=states)
        actions, logps = self._get_actions_impl(states, exploring, require_logps)
        assert self._shape_check(states=states, actions=actions)
        return actions, logps

    def _shape_check(self, states: torch.Tensor, actions: Optional[torch.Tensor] = None) -> bool:
        if not SHAPE_CHECK_FLAG:
            return True
        else:
            if states.shape[0] == 0:
                return False
            if not match_shape(states, (None, self.state_dim)):
                return False

            if actions is not None:
                if not match_shape(actions, (states.shape[0], self.action_dim)):
                    return False
            return True

    @abstractmethod
    def _get_actions_impl(
        self, states: torch.Tensor, exploring: bool, require_logps: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass

    def freeze_all_parameters(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all_parameters(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    @abstractmethod
    def freeze(self) -> None:
        pass

    @abstractmethod
    def unfreeze(self) -> None:
        pass


class DiscretePolicyNet(PolicyNet, metaclass=ABCMeta):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscretePolicyNet, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    @property
    def action_num(self) -> int:
        return self._action_num


class ContinuousPolicyNet(PolicyNet, metaclass=ABCMeta):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(ContinuousPolicyNet, self).__init__(state_dim=state_dim, action_dim=action_dim)
