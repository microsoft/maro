# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from maro.rl_v3.learning import ExpElement
from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.utils import CoroutineWrapper, RemoteObj

from .replay_memory import MultiReplayMemory
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
        self._agent2policy = {}

        self._dispatcher_address: Optional[Tuple[str, int]] = None
        print(f"Creating trainer {self.__class__.__name__} {self._name}.")

    @property
    def name(self) -> str:
        return self._name

    @property
    def agent_num(self) -> int:
        return len(self._agent2policy)

    def register_agent2policy(self, agent2policy: Dict[Any, str]) -> None:
        self._agent2policy = {
            agent_name: policy_name
            for agent_name, policy_name in agent2policy.items()
            if extract_trainer_name(policy_name) == self.name
        }

    @abstractmethod
    def register_policy_creator(
        self,
        global_policy_creator: Dict[str, Callable[[str], RLPolicy]]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def train_step(self) -> None:
        """Run a training step to update all the policies that this trainer is responsible for.
        """
        raise NotImplementedError

    @abstractmethod
    def build(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def record(self, exp_element: ExpElement) -> None:
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

    @abstractmethod
    async def get_policy_state(self) -> Dict[str, object]:
        """Get policies' states.

        Returns:
            A double-deck dict with format: {policy_name: policy_state}.
        """
        raise NotImplementedError


class SingleTrainer(AbsTrainer, metaclass=ABCMeta):
    """Policy trainer that trains only one policy.
    """
    def __init__(self, name: str, params: TrainerParams) -> None:
        super(SingleTrainer, self).__init__(name, params)

        self._ops: Union[RemoteObj, CoroutineWrapper, None] = None  # To be created in `build()`

        self._policy_creator: Dict[str, Callable[[str], RLPolicy]] = {}
        self._policy_name: Optional[str] = None
        self._get_policy_func: Optional[Callable] = None

    def register_policy_creator(
        self,
        global_policy_creator: Dict[str, Callable[[str], RLPolicy]]
    ) -> None:
        self._policy_creator: Dict[str, Callable[[str], RLPolicy]] = {
            policy_name: func for policy_name, func in global_policy_creator.items()
            if extract_trainer_name(policy_name) == self.name
        }

        if len(self._policy_creator) == 0:
            raise ValueError(f"Trainer {self._name} has no policies")
        if len(self._policy_creator) > 1:
            raise ValueError(f"Trainer {self._name} cannot have more than one policy assigned to it")

        self._policy_name = list(self._policy_creator.keys())[0]
        self._get_policy_func = lambda: self._policy_creator[self._policy_name](self._policy_name)

    async def get_policy_state(self) -> Dict[str, object]:
        if not self._ops:
            raise ValueError("'build' needs to be called to create an ops instance first.")
        policy_name, state = await self._ops.get_policy_state()
        return {policy_name: state}


class MultiTrainer(AbsTrainer, metaclass=ABCMeta):
    """Policy trainer that trains multiple policies.
    """
    def __init__(self, name: str, params: TrainerParams) -> None:
        super(MultiTrainer, self).__init__(name, params)
        self._policy_creator: Dict[str, Callable[[str], RLPolicy]] = {}
        self._policy_names: List[str] = []

    def register_policy_creator(
        self,
        global_policy_creator: Dict[str, Callable[[str], RLPolicy]]
    ) -> None:
        self._policy_creator: Dict[str, Callable[[str], RLPolicy]] = {
            policy_name: func for policy_name, func in global_policy_creator.items()
            if extract_trainer_name(policy_name) == self.name
        }
        self._policy_names = sorted(list(self._policy_creator.keys()))

    @abstractmethod
    async def get_policy_state(self) -> Dict[str, object]:
        raise NotImplementedError
