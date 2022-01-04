# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.utils import AbsTransitionBatch, MultiTransitionBatch, TransitionBatch
from maro.rl_v3.utils.distributed import RemoteObj

from .replay_memory import MultiReplayMemory, ReplayMemory


@dataclass
class TrainerParams:
    replay_memory_capacity: int = 10000
    batch_size: int = 128


class AbsTrainer(object, metaclass=ABCMeta):
    """Policy trainer used to train policies. Trainer maintains several train ops and
    controls training logics of them, while train ops take charge of specific policy updating.
    """
    def __init__(
        self,
        name: str,
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        params: TrainerParams,
    ) -> None:
        """
        Args:
            name (str): Name of the trainer.
            get_policy_func_dict (Dict[str, Callable[[str], RLPolicy]]): Dict of functions that used to create policies.
        """
        self._name = name
        self._get_policy_func_dict = get_policy_func_dict
        self._params = params
        print(f"Creating trainer {self.__class__.__name__} {name}.")

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def train_step(self) -> None:
        """Run a training step to update all the policies that this trainer is responsible for.
        """
        raise NotImplementedError

    @abstractmethod
    def get_policy_state_dict(self) -> Dict[str, object]:
        """Get policies' states.

        Returns:
            A double-deck dict with format: {policy_name: policy_state}.
        """
        raise NotImplementedError

    @abstractmethod
    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        """Set policies' states.

        Args:
            policy_state_dict (Dict[str, object]): A double-deck dict with format: {policy_name: policy_state}.
        """
        raise NotImplementedError

    @abstractmethod
    def local(self) -> None:
        # Create necessary ops for local training
        raise NotImplementedError

    @abstractmethod
    def create_local_ops(self, name: str = None):
        raise NotImplementedError

    def remote(self, dispatcher_address: Tuple[str, int]):
        pass


class SingleTrainer(AbsTrainer, metaclass=ABCMeta):
    """Policy trainer that trains only one policy."""
    def __init__(
        self,
        name: str,
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        params: TrainerParams
    ) -> None:
        if len(get_policy_func_dict) > 1:
            raise ValueError(f"trainer {self._name} cannot have more than one policy assigned to it")

        super(SingleTrainer, self).__init__(name, get_policy_func_dict, params)

        self._replay_memory: Optional[ReplayMemory] = None
        self._policy_name = list(get_policy_func_dict.keys())[0]
        self._get_policy_func = get_policy_func_dict[self._policy_name]
        self._ops = None

    def record(self, transition_batch: TransitionBatch) -> None:
        """Record the experiences collected by external modules.

        Args:
            transition_batch (TransitionBatch): A TransitionBatch item that contains a batch of experiences.
        """
        assert isinstance(transition_batch, TransitionBatch)
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._params.batch_size)

    def get_policy_state_dict(self) -> Dict[str, object]:
        if not self._ops:
            raise ValueError("'create_ops' needs to be called to create an ops instance first.")
        return {self._ops.policy_name: self._ops.get_policy_state()}

    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        if not self._ops:
            raise ValueError("'create_ops' needs to be called to create an ops instance first.")
        assert len(policy_state_dict) == 1 and self._ops.policy_name in policy_state_dict
        self._ops.set_policy_state(policy_state_dict[self._ops.policy_name])

    def remote(self, dispatcher_address: Tuple[str, int]):
        self._ops = RemoteObj(f"{self._policy_name}_ops", dispatcher_address)


class MultiTrainer(AbsTrainer, metaclass=ABCMeta):
    """Policy trainer that trains multiple policies.
    """
    def __init__(
        self,
        name: str,
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        params: TrainerParams
    ) -> None:
        super(MultiTrainer, self).__init__(name, get_policy_func_dict, params)

        self._get_policy_func_dict = get_policy_func_dict
        self._replay_memory: Optional[MultiReplayMemory] = None
        self._policy_names = sorted(list(get_policy_func_dict.keys()))
        self._ops_dict = {}

    @property
    def num_policies(self) -> int:
        return len(self._ops_dict)

    def record(self, transition_batch: MultiTransitionBatch) -> None:
        """Record the experiences collected by external modules.

        Args:
            transition_batch (MultiTransitionBatch): A TransitionBatch item that contains a batch of experiences.
        """
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> MultiTransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._params.batch_size)

    def get_policy_state_dict(self) -> Dict[str, object]:
        if len(self._ops_dict) == 0:
            raise ValueError("'create_ops' needs to be called to create an ops instance first.")

        return {name: ops.get_policy_state() for name, ops in self._ops_dict}

    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        if len(self._ops_dict) == 0:
            raise ValueError("'create_ops' needs to be called to create an ops instance first.")

        assert len(policy_state_dict) == len(self._ops_dict)
        for ops in self._ops_dict.values():
            ops.set_policy_state(policy_state_dict[ops.policy_name])

    def remote(self, dispatcher_address: Tuple[str, int]):
        self._ops_dict = {f"{name}_ops": RemoteObj(name, dispatcher_address) for name in self._get_policy_func_dict}


class BatchTrainer:
    def __init__(self, trainers: List[AbsTrainer]) -> None:
        self._trainers = trainers
        self._trainer_dict = {trainer.name: trainer for trainer in self._trainers}

    def record(self, batch_by_trainer: Dict[str, AbsTransitionBatch]) -> None:
        for trainer_name, batch in batch_by_trainer.items():
            self._trainer_dict[trainer_name].record(batch)

    def train(self) -> None:
        asyncio.run(self._train_impl())

    async def _train_impl(self) -> None:
        await asyncio.gather(*[trainer.train_step() for trainer in self._trainers])
