# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import collections
import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from maro.rl.policy import AbsPolicy, RLPolicy
from maro.rl.rollout import ExpElement
from maro.rl.utils import TransitionBatch
from maro.rl.utils.objects import FILE_SUFFIX
from maro.utils import LoggerV2

from .replay_memory import ReplayMemory
from .train_ops import AbsTrainOps, RemoteOps


@dataclass
class TrainerParams:
    """Common trainer parameters.

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
    reward_discount (float, default=0.9): Reward decay as defined in standard RL terminology.

    """

    replay_memory_capacity: int = 10000
    batch_size: int = 128
    data_parallelism: int = 1
    reward_discount: float = 0.9

    @abstractmethod
    def extract_ops_params(self) -> Dict[str, object]:
        """Extract parameters that should be passed to the train ops.

        Returns:
            params (Dict[str, object]): Parameter dict.
        """
        raise NotImplementedError


class AbsTrainer(object, metaclass=ABCMeta):
    """Policy trainer used to train policies. Trainer maintains a group of train ops and
    controls training logics of them, while train ops take charge of specific policy updating.

    Trainer will hold one or more replay memories to store the experiences, and it will also maintain a duplication
    of all policies it trains. However, trainer will not do any actual computations. All computations will be
    done in the train ops.

    Args:
        name (str): Name of the trainer.
        params (TrainerParams): Trainer's parameters.
    """

    def __init__(self, name: str, params: TrainerParams) -> None:
        self._name = name
        self._params = params
        self._batch_size = self._params.batch_size
        self._agent2policy: Dict[Any, str] = {}
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

    def register_agent2policy(self, agent2policy: Dict[Any, str], policy_trainer_mapping: Dict[str, str]) -> None:
        """Register the agent to policy dict that correspond to the current trainer. A valid policy name should start
        with the name of its trainer. For example, "DQN.POLICY_NAME". Therefore, we could identify which policies
        should be registered to the current trainer according to the policy's name.

        Args:
            agent2policy (Dict[Any, str]): Agent name to policy name mapping.
            policy_trainer_mapping (Dict[str, str]): Policy name to trainer name mapping.
        """
        self._agent2policy = {
            agent_name: policy_name
            for agent_name, policy_name in agent2policy.items()
            if policy_trainer_mapping[policy_name] == self.name
        }

    @abstractmethod
    def register_policy_creator(
        self,
        global_policy_creator: Dict[str, Callable[[], AbsPolicy]],
        policy_trainer_mapping: Dict[str, str],
    ) -> None:
        """Register the policy creator. Only keep the creators of the policies that the current trainer need to train.

        Args:
            global_policy_creator (Dict[str, Callable[[], AbsPolicy]]): Dict that contains the creators for all
                policies.
            policy_trainer_mapping (Dict[str, str]): Policy name to trainer name mapping.
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
        """Run a training step to update all the policies that this trainer is responsible for."""
        raise NotImplementedError

    async def train_step_as_task(self) -> None:
        """Update all policies managed by the trainer as an asynchronous task."""
        raise NotImplementedError

    @abstractmethod
    def record_multiple(self, env_idx: int, exp_elements: List[ExpElement]) -> None:
        """Record rollout all experiences from an environment in the replay memory.

        Args:
            env_idx (int): The index of the environment that generates this batch of experiences. This is used
                when there are more than one environment collecting experiences in parallel.
            exp_elements (List[ExpElement]): Experiences.
        """
        raise NotImplementedError

    def set_proxy_address(self, proxy_address: Tuple[str, int]) -> None:
        self._proxy_address = proxy_address

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


class SingleAgentTrainer(AbsTrainer, metaclass=ABCMeta):
    """Policy trainer that trains only one policy."""

    def __init__(self, name: str, params: TrainerParams) -> None:
        super(SingleAgentTrainer, self).__init__(name, params)
        self._policy_name: Optional[str] = None
        self._policy_creator: Optional[Callable[[], RLPolicy]] = None
        self._ops: Optional[AbsTrainOps] = None
        self._replay_memory: Optional[ReplayMemory] = None

    @property
    def ops(self):
        return self._ops

    def register_policy_creator(
        self,
        global_policy_creator: Dict[str, Callable[[], AbsPolicy]],
        policy_trainer_mapping: Dict[str, str],
    ) -> None:
        policy_names = [
            policy_name for policy_name in global_policy_creator if policy_trainer_mapping[policy_name] == self.name
        ]
        if len(policy_names) != 1:
            raise ValueError(f"Trainer {self._name} should have exactly one policy assigned to it")

        self._policy_name = policy_names.pop()
        self._policy_creator = global_policy_creator[self._policy_name]

    @abstractmethod
    def get_local_ops(self) -> AbsTrainOps:
        """Create an `AbsTrainOps` instance associated with the policy.

        Returns:
            ops (AbsTrainOps): The local ops.
        """
        raise NotImplementedError

    def get_ops(self) -> Union[RemoteOps, AbsTrainOps]:
        """Create an `AbsTrainOps` instance associated with the policy. If a proxy address has been registered to the
        trainer, this returns a `RemoteOps` instance in which all methods annotated as "remote" are turned into a remote
        method call. Otherwise, a regular `AbsTrainOps` is returned.

        Returns:
            ops (Union[RemoteOps, AbsTrainOps]): The ops.
        """
        ops = self.get_local_ops()
        return RemoteOps(ops, self._proxy_address, logger=self._logger) if self._proxy_address else ops

    def get_policy_state(self) -> Dict[str, object]:
        self._assert_ops_exists()
        policy_name, state = self._ops.get_policy_state()
        return {policy_name: state}

    def load(self, path: str) -> None:
        self._assert_ops_exists()

        policy_state = torch.load(os.path.join(path, f"{self.name}_policy.{FILE_SUFFIX}"))
        non_policy_state = torch.load(os.path.join(path, f"{self.name}_non_policy.{FILE_SUFFIX}"))

        self._ops.set_state(
            {
                "policy": policy_state,
                "non_policy": non_policy_state,
            },
        )

    def save(self, path: str) -> None:
        self._assert_ops_exists()

        ops_state = self._ops.get_state()
        policy_state = ops_state["policy"]
        non_policy_state = ops_state["non_policy"]

        torch.save(policy_state, os.path.join(path, f"{self.name}_policy.{FILE_SUFFIX}"))
        torch.save(non_policy_state, os.path.join(path, f"{self.name}_non_policy.{FILE_SUFFIX}"))

    def record_multiple(self, env_idx: int, exp_elements: List[ExpElement]) -> None:
        agent_exp_pool = collections.defaultdict(list)
        for exp_element in exp_elements:
            for agent_name in exp_element.agent_names:
                agent_exp_pool[agent_name].append(
                    (
                        exp_element.agent_state_dict[agent_name],
                        exp_element.action_dict[agent_name],
                        exp_element.reward_dict[agent_name],
                        exp_element.terminal_dict[agent_name],
                        exp_element.next_agent_state_dict.get(agent_name, exp_element.agent_state_dict[agent_name]),
                    ),
                )

        for agent_name, exps in agent_exp_pool.items():
            transition_batch = TransitionBatch(
                states=np.vstack([exp[0] for exp in exps]),
                actions=np.vstack([exp[1] for exp in exps]),
                rewards=np.array([exp[2] for exp in exps]),
                terminals=np.array([exp[3] for exp in exps]),
                next_states=np.vstack([exp[4] for exp in exps]),
            )
            transition_batch = self._preprocess_batch(transition_batch)
            self._replay_memory.put(transition_batch)

    @abstractmethod
    def _preprocess_batch(self, transition_batch: TransitionBatch) -> TransitionBatch:
        raise NotImplementedError

    def _assert_ops_exists(self) -> None:
        if not self._ops:
            raise ValueError("'build' needs to be called to create an ops instance first.")

    async def exit(self) -> None:
        self._assert_ops_exists()
        if isinstance(self._ops, RemoteOps):
            await self._ops.exit()


class MultiAgentTrainer(AbsTrainer, metaclass=ABCMeta):
    """Policy trainer that trains multiple policies."""

    def __init__(self, name: str, params: TrainerParams) -> None:
        super(MultiAgentTrainer, self).__init__(name, params)
        self._policy_creator: Dict[str, Callable[[], RLPolicy]] = {}
        self._policy_names: List[str] = []
        self._ops_dict: Dict[str, AbsTrainOps] = {}

    @property
    def ops_dict(self):
        return self._ops_dict

    def register_policy_creator(
        self,
        global_policy_creator: Dict[str, Callable[[], AbsPolicy]],
        policy_trainer_mapping: Dict[str, str],
    ) -> None:
        self._policy_creator: Dict[str, Callable[[], RLPolicy]] = {
            policy_name: func
            for policy_name, func in global_policy_creator.items()
            if policy_trainer_mapping[policy_name] == self.name
        }
        self._policy_names = list(self._policy_creator.keys())

    @abstractmethod
    def get_local_ops(self, name: str) -> AbsTrainOps:
        """Create an `AbsTrainOps` instance with a given name.

        Args:
            name (str): Ops name.

        Returns:
            ops (AbsTrainOps): The local ops.
        """
        raise NotImplementedError

    def get_ops(self, name: str) -> Union[RemoteOps, AbsTrainOps]:
        """Create an `AbsTrainOps` instance with a given name. If a proxy address has been registered to the trainer,
        this returns a `RemoteOps` instance in which all methods annotated as "remote" are turned into a remote method
        call. Otherwise, a regular `AbsTrainOps` is returned.

        Args:
            name (str): Ops name.

        Returns:
            ops (Union[RemoteOps, AbsTrainOps]): The ops.
        """
        ops = self.get_local_ops(name)
        return RemoteOps(ops, self._proxy_address, logger=self._logger) if self._proxy_address else ops

    @abstractmethod
    def get_policy_state(self) -> Dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    async def exit(self) -> None:
        raise NotImplementedError
