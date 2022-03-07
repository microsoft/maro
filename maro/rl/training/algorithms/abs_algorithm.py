# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from maro.rl.policy import RLPolicy
from maro.rl.rollout import ExpElement
from maro.utils import LoggerV2

from maro.rl.training.train_ops import AbsTrainOps, RemoteOps
from maro.rl.training.utils import extract_algo_inst_name


@dataclass
class AlgorithmParams:
    """Common algorithm parameters.

    device (str, default=None): Name of the device to store this algorithm instance. If it is None, the device will be
        automatically determined according to GPU availability.
    replay_memory_capacity (int, default=100000): Maximum capacity of the replay memory.
    batch_size (int, default=128): Training batch size.
    data_parallelism (int, default=1): Degree of data parallelism. A value greater than 1 can be used when
        a model is large and computing gradients with respect to a batch becomes expensive. In this case, the
        batch may be split into multiple smaller batches whose gradients can be computed in parallel on a set
        of remote nodes. For simplicity, only synchronous parallelism is supported, meaning that the model gets
        updated only after collecting all the gradients from the remote nodes. Note that this value is the desired
        parallelism and the actual parallelism in a distributed experiment may be smaller depending on the
        availability of compute resources. For details on distributed deep learning and data parallelism, see
        https://web.stanford.edu/~rezab/classes/cme323/S16/projects_reports/hedge_usmani.pdf, as well as an abundance
        of resources available on the internet.

    """
    device: str = None
    replay_memory_capacity: int = 10000
    batch_size: int = 128
    data_parallelism: int = 1

    @abstractmethod
    def extract_ops_params(self) -> Dict[str, object]:
        """Extract parameters that should be passed to the train ops.

        Returns:
            params (Dict[str, object]): Parameter dict.
        """
        raise NotImplementedError


class AbsAlgorithm(object, metaclass=ABCMeta):
    """RL algorithm used to train policies. An algorithm instance maintains a group of train ops and
    controls training logics of them, while train ops take charge of specific policy updating.

    An algorithm instance will hold one or more replay memories to store the experiences, and it will also maintain
    a duplication of all policies it trains. However, algorithms instances will not do any actual computations.
    All computations will be done in the train ops.

    Args:
        name (str): Name of the algorithm instance.
        params (AlgorithmParams): Algorithm parameters.
    """

    def __init__(self, name: str, params: AlgorithmParams) -> None:
        self._name = name
        self._batch_size = params.batch_size
        self._agent2policy: Dict[str, str] = {}
        self._proxy_address: Optional[Tuple[str, int]] = None
        self._logger = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def agent_num(self) -> int:
        return len(self._agent2policy)

    def register_logger(self, logger: LoggerV2) -> None:
        self._logger = logger

    def register_agent2policy(self, agent2policy: Dict[Any, str]) -> None:
        """Register the agent to policy dict that correspond to the current algorithm instance. A valid policy name
        should start with the name of its algorithm instance. For example, "DQN.POLICY_NAME". Therefore, we could
        identify which policies should be registered to the current algorithm instance according to the policy's name.

        Args:
            agent2policy (Dict[Any, str]): Agent name to policy name mapping.
        """
        self._agent2policy = {
            agent_name: policy_name
            for agent_name, policy_name in agent2policy.items()
            if extract_algo_inst_name(policy_name) == self.name
        }

    @abstractmethod
    def register_policy_creator(
        self,
        global_policy_creator: Dict[str, Callable[[str], RLPolicy]],
    ) -> None:
        """Register the policy creator. Only keep the creators of the policies that the current algorithm instance
        need to train.

        Args:
            global_policy_creator (Dict[str, Callable[[str], RLPolicy]]): Dict that contains the creators for all
                policies.
        """
        raise NotImplementedError

    @abstractmethod
    def build(self) -> None:
        """Create the required train-ops and replay memory. This should be called before invoking `train` or
        `train_as_task`.
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(self) -> None:
        """Run a training step to update all the policies that this algorithm instance is responsible for.
        """
        raise NotImplementedError

    async def train_step_as_task(self) -> None:
        """Update all policies managed by the algorithm instance as an asynchronous task.
        """
        raise NotImplementedError

    @abstractmethod
    def record(self, env_idx: int, exp_element: ExpElement) -> None:
        """Record rollout experiences in the replay memory.

        Args:
            env_idx (int): The index of the environment that generates this batch of experiences. This is used
                when there are more than one environment collecting experiences in parallel.
            exp_element (ExpElement): Experiences.
        """
        raise NotImplementedError

    def set_proxy_address(self, proxy_address: Tuple[str, int]) -> None:
        self._proxy_address = proxy_address

    @abstractmethod
    def get_local_ops_by_name(self, name: str) -> AbsTrainOps:
        """Create an `AbsTrainOps` instance with a given name.

        Args:
            name (str): Ops name.

        Returns:
            ops (AbsTrainOps): The local ops.
        """
        raise NotImplementedError

    def get_ops(self, name: str) -> Union[RemoteOps, AbsTrainOps]:
        """Create an `AbsTrainOps` instance with a given name. If a proxy address has been registered to the algorithm
        instance, this returns a `RemoteOps` instance in which all methods annotated as "remote" are turned into a
        remote method call. Otherwise, a regular `AbsTrainOps` is returned.

        Args:
            name (str): Ops name.

        Returns:
            ops (Union[RemoteOps, AbsTrainOps]): The ops.
        """
        ops = self.get_local_ops_by_name(name)
        return RemoteOps(ops, self._proxy_address, logger=self._logger) if self._proxy_address else ops

    @abstractmethod
    def get_policy_state(self) -> Dict[str, object]:
        """Get policies' states.

        Returns:
            A double-deck dict with format: {policy_name: policy_state}.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def exit(self) -> None:
        raise NotImplementedError


class SingleAlgorithm(AbsAlgorithm, metaclass=ABCMeta):
    """Algorithm that trains only one policy.
    """

    def __init__(self, name: str, params: AlgorithmParams) -> None:
        super(SingleAlgorithm, self).__init__(name, params)

        self._ops: Union[RemoteOps, None] = None  # To be created in `build()`

        self._policy_creator: Dict[str, Callable[[str], RLPolicy]] = {}
        self._policy_name: Optional[str] = None
        self._get_policy_func: Optional[Callable] = None

    def register_policy_creator(
        self,
        global_policy_creator: Dict[str, Callable[[str], RLPolicy]],
    ) -> None:
        self._policy_creator: Dict[str, Callable[[str], RLPolicy]] = {
            policy_name: func for policy_name, func in global_policy_creator.items()
            if extract_algo_inst_name(policy_name) == self.name
        }

        if len(self._policy_creator) == 0:
            raise ValueError(f"Algorithm instance {self._name} has no policies")
        if len(self._policy_creator) > 1:
            raise ValueError(f"Algorithm instance {self._name} cannot have more than one policy assigned to it")

        self._policy_name = list(self._policy_creator.keys())[0]
        self._get_policy_func = lambda: self._policy_creator[self._policy_name](self._policy_name)

    def get_policy_state(self) -> Dict[str, object]:
        self._assert_ops_exists()
        policy_name, state = self._ops.get_policy_state()
        return {policy_name: state}

    def load(self, path: str) -> None:
        self._assert_ops_exists()
        self._ops.set_state(torch.load(path))

    def save(self, path: str) -> None:
        self._assert_ops_exists()
        torch.save(self._ops.get_state(), path)

    def _assert_ops_exists(self) -> None:
        if not self._ops:
            raise ValueError("'build' needs to be called to create an ops instance first.")

    async def exit(self) -> None:
        if isinstance(self._ops, RemoteOps):
            await self._ops.exit()


class MultiAlgorithm(AbsAlgorithm, metaclass=ABCMeta):
    """Algorithm that trains multiple policies.
    """

    def __init__(self, name: str, params: AlgorithmParams) -> None:
        super(MultiAlgorithm, self).__init__(name, params)
        self._policy_creator: Dict[str, Callable[[str], RLPolicy]] = {}
        self._policy_names: List[str] = []

    def register_policy_creator(
        self,
        global_policy_creator: Dict[str, Callable[[str], RLPolicy]],
    ) -> None:
        self._policy_creator: Dict[str, Callable[[str], RLPolicy]] = {
            policy_name: func for policy_name, func in global_policy_creator.items()
            if extract_algo_inst_name(policy_name) == self.name
        }
        self._policy_names = sorted(list(self._policy_creator.keys()))

    @abstractmethod
    def get_policy_state(self) -> Dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    async def exit(self) -> None:
        raise NotImplementedError
