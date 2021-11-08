from abc import abstractmethod
from typing import Dict, Optional

from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.utils.transition_batch import TransitionBatch


class AbsTrainer(object):
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


class SingleTrainer(AbsTrainer):
    def __init__(self, name: str) -> None:
        super(SingleTrainer, self).__init__(name)
        self._policy: Optional[RLPolicy] = None

    def record(
        self,
        policy_name: str,  # TODO: need this?
        transition_batch: TransitionBatch
    ) -> None:
        self._record_impl(
            policy_name=policy_name,
            transition_batch=transition_batch
        )

    @abstractmethod
    def _record_impl(self, policy_name: str, transition_batch: TransitionBatch) -> None:
        pass

    @abstractmethod
    def register_policy(self, policy: RLPolicy) -> None:
        pass

    def get_policy_state_dict(self) -> Dict[str, object]:
        return {self._policy.name: self._policy.get_policy_state()}

    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        assert len(policy_state_dict) == 1 and self._policy.name in policy_state_dict
        self._policy.set_policy_state(policy_state_dict[self._policy.name])
