# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.utils import AbsTransitionBatch, CoroutineWrapper, MultiTransitionBatch, RemoteObj, TransitionBatch

from .replay_memory import MultiReplayMemory, ReplayMemory
from .train_ops import AbsTrainOps
from .utils import extract_trainer_name


@dataclass
class TrainerParams:
    device: str = None
    enable_data_parallelism: bool = False
    replay_memory_capacity: int = 10000
    batch_size: int = 128

    @abstractmethod
    def extract_ops_params(self) -> Dict[str, object]:
        raise NotImplementedError


class AbsTrainer(object, metaclass=ABCMeta):
    """Policy trainer used to train policies. Trainer maintains several train ops and
    controls training logics of them, while train ops take charge of specific policy updating.
    """
    def __init__(self, name: str, params: TrainerParams) -> None:
        self._name = name
        self._batch_size = params.batch_size

        self._dispatcher_address: Optional[Tuple[str, int]] = None
        print(f"Creating trainer {self.__class__.__name__} {self._name}.")

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def register_get_policy_func_dict(
        self,
        global_get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def train_step(self) -> None:
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
    def build(self) -> None:
        raise NotImplementedError

    def set_dispatch_address(self, dispatcher_address: Tuple[str, int]) -> None:
        self._dispatcher_address = dispatcher_address

    @abstractmethod
    def _get_local_ops_by_name(self, ops_name: str) -> AbsTrainOps:
        raise NotImplementedError

    def get_ops(self, ops_name: str) -> Union[RemoteObj, CoroutineWrapper]:
        if self._dispatcher_address:
            return RemoteObj(ops_name, self._dispatcher_address)
        else:
            return CoroutineWrapper(self._get_local_ops_by_name(ops_name))


class SingleTrainer(AbsTrainer, metaclass=ABCMeta):
    """Policy trainer that trains only one policy.
    """
    def __init__(self, name: str, params: TrainerParams) -> None:
        super(SingleTrainer, self).__init__(name, params)

        self._ops: Union[RemoteObj, CoroutineWrapper, None] = None  # To be created in `build()`
        self._replay_memory: Optional[ReplayMemory] = None  # To be created in `build()`

    def register_get_policy_func_dict(
        self,
        global_get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]]
    ) -> None:
        self._get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]] = {
            policy_name: func for policy_name, func in global_get_policy_func_dict.items()
            if extract_trainer_name(policy_name) == self.name
        }

        if len(self._get_policy_func_dict) == 0:
            raise ValueError(f"Trainer {self._name} has no policies")
        if len(self._get_policy_func_dict) > 1:
            raise ValueError(f"Trainer {self._name} cannot have more than one policy assigned to it")

        self._policy_name = list(self._get_policy_func_dict.keys())[0]
        self._get_policy_func = lambda: self._get_policy_func_dict[self._policy_name](self._policy_name)

    def record(self, transition_batch: TransitionBatch) -> None:
        """Record the experiences collected by external modules.

        Args:
            transition_batch (TransitionBatch): A TransitionBatch item that contains a batch of experiences.
        """
        assert isinstance(transition_batch, TransitionBatch)
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._batch_size)

    def get_policy_state_dict(self) -> Dict[str, object]:
        if not self._ops:
            raise ValueError("'create_ops' needs to be called to create an ops instance first.")
        return {self._ops.policy_name: self._ops.get_policy_state()}


class MultiTrainer(AbsTrainer, metaclass=ABCMeta):
    """Policy trainer that trains multiple policies.
    """
    def __init__(self, name: str, params: TrainerParams) -> None:
        super(MultiTrainer, self).__init__(name, params)

        self._replay_memory: Optional[MultiReplayMemory] = None  # To be created in `build()`

    def register_get_policy_func_dict(
        self,
        global_get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]]
    ) -> None:
        self._get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]] = {
            policy_name: func for policy_name, func in global_get_policy_func_dict.items()
            if extract_trainer_name(policy_name) == self.name
        }
        self._policy_names = sorted(list(self._get_policy_func_dict.keys()))

    @property
    def num_policies(self) -> int:
        return len(self._policy_names)

    def record(self, transition_batch: MultiTransitionBatch) -> None:
        """Record the experiences collected by external modules.

        Args:
            transition_batch (MultiTransitionBatch): A TransitionBatch item that contains a batch of experiences.
        """
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> MultiTransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._batch_size)


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
