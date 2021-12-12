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
    def __init__(self, name: str, device: str = None, enable_data_parallelism: bool = False) -> None:
        """
        Args:
            name (str): Name of the trainer
            device (str): Device to store this trainer. If it is None, the device will be set to "cpu" if cuda is
                unavailable and "cuda" otherwise. Defaults to None.
            enable_data_parallelism (bool): Whether to enable data parallelism in this trainer.
        """
        self._name = name
        self._device = torch.device(device) if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Creating trainer {self.__class__.__name__} {name} on device {self._device}")

        self._enable_data_parallelism = enable_data_parallelism

    @property
    def name(self) -> str:
        return self._name

    def train_step(self) -> None:
        self._train_step_impl()

    @abstractmethod
    def _train_step_impl(self) -> None:
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
    def __init__(
        self,
        name: str,
        device: str = None,
        enable_data_parallelism: bool = False,
        train_batch_size: int = 128
    ) -> None:
        super(SingleTrainer, self).__init__(name, device, enable_data_parallelism)
        self._policy_name: Optional[str] = None
        self._replay_memory = Optional[ReplayMemory]
        self._train_batch_size = train_batch_size

    def record(
        self,
        transition_batch: TransitionBatch
    ) -> None:
        """
        Record the experiences collected by external modules.

        Args:
            transition_batch (TransitionBatch): A TransitionBatch item that contains a batch of experiences.
        """
        self._record_impl(transition_batch=transition_batch)

    def _record_impl(self, transition_batch: TransitionBatch) -> None:
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def register_policy(self, policy: RLPolicy) -> None:
        """
        Register the policy and finish other related initializations.
        """
        self._policy_name = policy.name
        self._register_policy_impl(policy)

    @abstractmethod
    def _register_policy_impl(self, policy: RLPolicy) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_policy_state(self) -> object:
        raise NotImplementedError

    @abstractmethod
    def set_policy_state(self, policy_state: object) -> None:
        raise NotImplementedError

    def get_policy_state_dict(self) -> Dict[str, object]:
        return {self._policy_name: self.get_policy_state()}

    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        assert len(policy_state_dict) == 1 and self._policy_name in policy_state_dict
        self.set_policy_state(policy_state_dict[self._policy_name])


class MultiTrainer(AbsTrainer, metaclass=ABCMeta):
    """
    Policy trainer that trains multiple policies.
    """

    def __init__(
        self,
        name: str,
        device: str = None,
        enable_data_parallelism: bool = False,
        train_batch_size: int = 128
    ) -> None:
        super(MultiTrainer, self).__init__(name, device, enable_data_parallelism)
        self._policy_names: List[str] = []
        self._replay_memory: Optional[MultiReplayMemory] = None
        self._train_batch_size = train_batch_size

    @property
    def num_policies(self) -> int:
        return len(self._policy_names)

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

    def _record_impl(self, transition_batch: MultiTransitionBatch) -> None:
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> MultiTransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def register_policies(self, policies: List[RLPolicy]) -> None:
        """
        Register the policies and finish other related initializations.
        """
        self._policy_names = [policy.name for policy in policies]
        self._register_policies_impl(policies)

    @abstractmethod
    def _register_policies_impl(self, policies: List[RLPolicy]) -> None:
        raise NotImplementedError
