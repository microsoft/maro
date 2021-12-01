from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional

import torch

from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.replay_memory import MultiReplayMemory, ReplayMemory
from maro.rl_v3.utils import MultiTransitionBatch, TransitionBatch


class AbsTrainer(object, metaclass=ABCMeta):
    """
    Policy trainer used to train policies.
    """
    def __init__(self, name: str, device: str = None) -> None:
        self._name = name
        self._device = torch.device(device) if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Creating trainer {self.__class__.__name__} {name} on device {self._device}")

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def train_step(self) -> None:
        """
        Run a training step to update all the policies that this trainer is responsible for.
        """
        raise NotImplementedError

    @abstractmethod
    def get_policy_state_dict(self) -> Dict[str, object]:
        """
        Get policies' states.

        Returns:
            A double-deck dict with format: {policy_name: policy_state}.
        """
        raise NotImplementedError

    @abstractmethod
    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        """
        Set policies' states.

        Args:
            policy_state_dict (Dict[str, object]): A double-deck dict with format: {policy_name: policy_state}.
        """
        raise NotImplementedError


class SingleTrainer(AbsTrainer, metaclass=ABCMeta):
    """
    Policy trainer that trains only one policy.
    """
    def __init__(self, name: str, device: str = None) -> None:
        super(SingleTrainer, self).__init__(name, device)
        self._policy: Optional[RLPolicy] = None
        self._replay_memory = Optional[ReplayMemory]

    def record(
        self,
        policy_name: str,  # TODO: need this?
        transition_batch: TransitionBatch
    ) -> None:
        """
        Record the experiences collected by external modules.

        Args:
            policy_name (str): The name of the policy that generates this batch.
            transition_batch (TransitionBatch): A TransitionBatch item that contains a batch of experiences.
        """
        self._record_impl(
            policy_name=policy_name,
            transition_batch=transition_batch
        )

    def _record_impl(self, policy_name: str, transition_batch: TransitionBatch) -> None:
        """
        Implementation of `record`.
        """
        self._replay_memory.put(transition_batch)

    def register_policy(self, policy: RLPolicy) -> None:
        """
        Register the policy and finish other related initializations.
        """
        policy.to_device(self._device)
        self._register_policy_impl(policy)

    @abstractmethod
    def _register_policy_impl(self, policy: RLPolicy) -> None:
        raise NotImplementedError

    def get_policy_state_dict(self) -> Dict[str, object]:
        return {self._policy.name: self._policy.get_policy_state()}

    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        assert len(policy_state_dict) == 1 and self._policy.name in policy_state_dict
        self._policy.set_policy_state(policy_state_dict[self._policy.name])


class MultiTrainer(AbsTrainer, metaclass=ABCMeta):
    """
    Policy trainer that trains multiple policies.
    """
    def __init__(self, name: str, device: str = None) -> None:
        super(MultiTrainer, self).__init__(name, device)
        self._policy_dict: Dict[str, RLPolicy] = {}
        self._policies: List[RLPolicy] = []
        self._replay_memory: Optional[MultiReplayMemory] = None

    @property
    def num_policies(self):
        return len(self._policies)

    def record(
        self,
        transition_batch: MultiTransitionBatch
    ) -> None:
        """
        Record the experiences collected by external modules.

        Args:
            transition_batch (MultiTransitionBatch): A TransitionBatch item that contains a batch of experiences.
        """
        self._record_impl(transition_batch)

    @abstractmethod
    def _record_impl(self, transition_batch: MultiTransitionBatch) -> None:
        raise NotImplementedError

    def register_policies(self, policies: List[RLPolicy]) -> None:
        for policy in policies:
            policy.to_device(self._device)
        self._register_policies_impl(policies)

    @abstractmethod
    def _register_policies_impl(self, policies: List[RLPolicy]) -> None:
        pass

    def get_policy_state_dict(self) -> Dict[str, object]:
        return {policy_name: policy.get_policy_state() for policy_name, policy in self._policy_dict.items()}

    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        assert len(policy_state_dict) == len(self._policy_dict)
        for policy_name, policy_state in policy_state_dict.items():
            assert policy_name in self._policy_dict
            self._policy_dict[policy_name].set_policy_state(policy_state)
