import asyncio
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import torch

from maro.rl_v3.distributed.remote_ops import RemoteOps
from maro.rl_v3.replay_memory import MultiReplayMemory, ReplayMemory
from maro.rl_v3.utils import AbsTransitionBatch, MultiTransitionBatch, TransitionBatch


class AbsTrainer(object, metaclass=ABCMeta):
    """Policy trainer used to train policies. Trainer maintains several train workers and
    controls training logics of them, while train workers take charge of specific policy updating.
    """
    def __init__(
        self,
        name: str,
        device: str = None,
        enable_data_parallelism: bool = False,
        train_batch_size: int = 128
    ) -> None:
        """
        Args:
            name (str): Name of the trainer
            device (str): Device to store this trainer. If it is None, the device will be set to "cpu" if cuda is
                unavailable and "cuda" otherwise. Defaults to None.
            enable_data_parallelism (bool): Whether to enable data parallelism in this trainer.
            train_batch_size (int): train batch size.
        """
        self._name = name
        self._device = torch.device(device) if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._enable_data_parallelism = enable_data_parallelism
        self._train_batch_size = train_batch_size

        print(f"Creating trainer {self.__class__.__name__} {name} on device {self._device}")

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    async def train_step(self) -> None:
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
        ops_creator: Dict[str, Callable],  # TODO
        dispatcher_address: Tuple[str, int] = None,
        device: str = None,
        enable_data_parallelism: bool = False,
        train_batch_size: int = 128
    ) -> None:
        super(SingleTrainer, self).__init__(name, device, enable_data_parallelism, train_batch_size)

        self._replay_memory: Optional[ReplayMemory] = None

        ops_names = [ops_name for ops_name in ops_creator if ops_name.startswith(f"{self._name}.")]
        if len(ops_names) > 1:
            raise ValueError(f"trainer {self._name} cannot have more than one policy assigned to it")

        ops_name = ops_names.pop()
        self._ops = RemoteOps(ops_name, dispatcher_address) if dispatcher_address else ops_creator[ops_name](ops_name)

    def record(self, transition_batch: TransitionBatch) -> None:
        """
        Record the experiences collected by external modules.

        Args:
            transition_batch (TransitionBatch): A TransitionBatch item that contains a batch of experiences.
        """
        assert isinstance(transition_batch, TransitionBatch)
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> TransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    def get_policy_state_dict(self) -> Dict[str, object]:
        if not self._ops:
            raise ValueError("'init_ops' needs to be called to create an ops instance first.")
        return {self._ops.name: self._ops.get_policy_state()}

    def set_policy_state_dict(self, policy_state_dict: Dict[str, object]) -> None:
        if not self._ops:
            raise ValueError("'init_ops' needs to be called to create an ops instance first.")
        assert len(policy_state_dict) == 1 and self._ops.name in policy_state_dict
        self._ops.set_policy_state(policy_state_dict[self._ops.name])


class MultiTrainer(AbsTrainer, metaclass=ABCMeta):
    """
    Policy trainer that trains multiple policies.
    """
    def __init__(
        self,
        name: str,
        ops_creator: Dict[str, Callable],
        dispatcher_address: Tuple[str, int] = None,
        device: str = None,
        enable_data_parallelism: bool = False,
        train_batch_size: int = 128
    ) -> None:
        super(MultiTrainer, self).__init__(name, device, enable_data_parallelism, train_batch_size)

        self._replay_memory: Optional[MultiReplayMemory] = None

        ops_names = [ops_name for ops_name in ops_creator if ops_name.startswith(f"{self._name}.")]
        # if len(ops_names) < 2:
        #     raise ValueError(f"trainer {self._name} cannot less than 2 policies assigned to it")

        self._ops_list = [
            RemoteOps(ops_name, dispatcher_address) if dispatcher_address else ops_creator[ops_name](ops_name)
            for ops_name in ops_names
        ]

    @property
    def num_policies(self) -> int:
        return len(self._ops_list)  # TODO

    def record(self, transition_batch: MultiTransitionBatch) -> None:
        """
        Record the experiences collected by external modules.

        Args:
            transition_batch (MultiTransitionBatch): A TransitionBatch item that contains a batch of experiences.
        """
        self._replay_memory.put(transition_batch)

    def _get_batch(self, batch_size: int = None) -> MultiTransitionBatch:
        return self._replay_memory.sample(batch_size if batch_size is not None else self._train_batch_size)

    async def parallelize(self, ops_func_name: str, *args, **kwargs):
        ret = [getattr(ops, ops_func_name)(*args, **kwargs) for ops in self._ops_list]
        return await asyncio.gather(*ret) if isinstance(self._ops_list[0], RemoteOps) else ret


class BatchTrainer:
    def __init__(self, trainers: List[AbsTrainer]) -> None:
        self._trainers = trainers
        self._trainer_dict = {trainer.name: trainer for trainer in self._trainers}

    def record(self, batch_by_trainer: Dict[str, AbsTransitionBatch]) -> None:
        for trainer_name, batch in batch_by_trainer.items():
            self._trainer_dict[trainer_name].record(batch)

    def train(self) -> None:
        asyncio.run(self._train_in_parallel())

    async def _train_in_parallel(self):
        await asyncio.gather(*[trainer.train_step() for trainer in self._trainers])
