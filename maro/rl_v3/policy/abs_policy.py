from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from itertools import count
from typing import Optional, Tuple

import numpy as np
import torch

from maro.rl_v3.policy_learner import AbsLearner
from maro.rl_v3.utils import match_shape


class AbsPolicy(object):
    _policy_counter = defaultdict(count)

    def __init__(self, name: str, trainable: bool) -> None:
        super(AbsPolicy, self).__init__()

        cls_name = self.__class__.__name__
        self._name = f"{cls_name}__{next(AbsPolicy._policy_counter[cls_name])}__{name}_{trainable}"
        self._trainable = trainable

    @abstractmethod
    def get_actions(self, states: object) -> object:
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def trainable(self) -> bool:
        return self._trainable


class DummyPolicy(AbsPolicy):
    def __init__(self) -> None:
        super(DummyPolicy, self).__init__(name='DUMMY_POLICY', trainable=False)

    def get_actions(self, states: object) -> object:
        return None


class RuleBasedPolicy(AbsPolicy, metaclass=ABCMeta):
    def __init__(self, name: str) -> None:
        super(RuleBasedPolicy, self).__init__(name=name, trainable=False)

    def get_actions(self, states: object) -> object:
        return self._rule(states)

    @abstractmethod
    def _rule(self, states: object) -> object:
        pass


class RLPolicy(AbsPolicy):
    def __init__(
        self,
        name: str,
        state_dim: int,
        action_dim: int,
        device: str,
        trainable: bool = True
    ) -> None:
        super(RLPolicy, self).__init__(name=name, trainable=trainable)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._device = torch.device(device) if device is not None \
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._exploring = False
        self._learner: Optional[AbsLearner] = None

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def explore(self) -> None:
        self._exploring = True

    def exploit(self) -> None:
        self._exploring = False

    def ndarray_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(self._device)

    @abstractmethod  # TODO
    def step(self, loss: torch.Tensor) -> None:
        pass

    @abstractmethod  # TODO
    def get_gradients(self, loss: torch.Tensor) -> torch.Tensor:
        pass

    def register_learner(self, algo: AbsLearner) -> None:
        self._learner = algo

    @property
    def learner(self) -> AbsLearner:
        return self._learner

    def get_actions(self, states: np.ndarray) -> np.ndarray:
        return self.get_actions_with_logps(states, require_logps=False)[0]

    def get_actions_tensor(self, states: torch.Tensor) -> torch.Tensor:
        return self.get_actions_with_logps_tensor(states, require_logps=False)[0]

    @abstractmethod
    def _get_actions_with_logps_impl(
        self, states: torch.Tensor, exploring: bool, require_logps: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass

    def get_actions_with_logps(
        self, states: np.ndarray, require_logps: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        actions, logps = self.get_actions_with_logps_tensor(self.ndarray_to_tensor(states), require_logps)
        return actions.cpu().numpy(), logps.cpu().numpy() if logps is not None else None

    def get_actions_with_logps_tensor(
        self, states: torch.Tensor, require_logps: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert self._shape_check(states=states)
        actions, logps = self._get_actions_with_logps_impl(states, self._exploring, require_logps)
        assert self._shape_check(states=states, actions=actions)  # [B, action_dim]
        assert logps is None or match_shape(logps, (states.shape[0],))  # [B]
        assert self._post_check(states=states, actions=actions)
        return actions, logps

    @abstractmethod
    def freeze(self) -> None:
        pass

    @abstractmethod
    def unfreeze(self) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def get_policy_state(self) -> object:
        pass

    @abstractmethod
    def set_policy_state(self, policy_state: object) -> None:
        pass

    @abstractmethod
    def soft_update(self, other_policy: RLPolicy, tau: float) -> None:
        pass

    def _shape_check(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> bool:
        if states.shape[0] == 0:
            return False
        if not match_shape(states, (None, self.state_dim)):
            return False

        if actions is not None:
            if not match_shape(actions, (states.shape[0], self.action_dim)):
                return False
        return True

    @abstractmethod
    def _post_check(self, states: torch.Tensor, actions: torch.Tensor) -> bool:
        pass


if __name__ == '__main__':
    data = [AbsPolicy('Jack', True), AbsPolicy('Tom', True), DummyPolicy(), DummyPolicy()]
    for policy in data:
        print(policy.name)
