from abc import abstractmethod
from typing import Dict, Optional

import numpy as np

from maro.rl_v3.policy import RLPolicy


class AbsLearner(object):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def train_step(self) -> None:
        pass

    @abstractmethod
    def get_policy_state_dict(self) -> Dict[str, object]:
        pass

    @abstractmethod
    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        pass


class SingleLearner(AbsLearner):
    def __init__(self, name: str) -> None:
        super(SingleLearner, self).__init__(name)
        self._policy: Optional[RLPolicy] = None

    def record(
        self,
        policy_name: str,  # TODO: need this?
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: bool,
        next_state: np.ndarray = None,
        value: float = None,
        logp: float = None
    ) -> None:
        assert len(state.shape) == 1 and state.shape[0] != 0
        assert len(action.shape) == 1 and action.shape[0] != 0
        assert next_state is None or state.shape == next_state.shape

        self._record_impl(
            policy_name=policy_name,
            states=np.expand_dims(state, axis=0),
            actions=np.expand_dims(action, axis=0),
            rewards=np.array([reward]),
            terminals=np.array([terminal]),
            next_states=None if next_state is None else np.expand_dims(next_state, axis=0),
            values=None if value is None else np.array([value]),
            logps=None if logp is None else np.array([logp])
        )

    @abstractmethod
    def _record_impl(
        self,
        policy_name: str,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        next_states: np.ndarray = None,
        values: np.ndarray = None,
        logps: np.ndarray = None
    ) -> None:
        pass

    @abstractmethod
    def register_policy(self, policy: RLPolicy) -> None:
        pass

    def get_policy_state_dict(self) -> Dict[str, object]:
        return {self._policy.name: self._policy.get_policy_state()}

    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        assert len(policy_state_dict) == 1 and self._policy.name in policy_state_dict
        self._policy.set_policy_state(policy_state_dict[self._policy.name])

#
#
# class MultiLearner(AbsLearner):
#     def __init__(self) -> None:
#         super(MultiLearner, self).__init__()
#
#     @abstractmethod
#     def record(
#         self,
#         policy_name: str,  # TODO: need this?
#         global_state: np.ndarray,
#         local_states: List[np.ndarray],
#         actions: List[np.ndarray],
#         rewards: List[float],
#         next_state: np.ndarray,
#         terminal: bool
#     ) -> None:
#         pass
