# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List

import torch
from tianshou.data import Batch

from maro.rl_v31.objects import ExpElement
from maro.rl_v31.policy.base import BaseRLPolicy
from maro.rl_v31.training.replay_memory import ReplayMemoryManager


class BaseTrainOps(object):
    def __init__(self, name: str, policy: BaseRLPolicy, reward_discount: float = 0.99) -> None:
        self.name = name
        self._policy = policy
        self._reward_discount = reward_discount

    def get_state(self) -> dict:
        return {
            "auxiliary": self._get_auxiliary_state(),
        }

    def set_state(self, ops_state_dict: dict) -> None:
        self._set_auxiliary_state(ops_state_dict["auxiliary"])

    @abstractmethod
    def _get_auxiliary_state(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _set_auxiliary_state(self, auxiliary_state: dict) -> None:
        raise NotImplementedError


class BaseTrainer(object, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        rmm: ReplayMemoryManager,
        batch_size: int = 128,
        reward_discount: float = 0.99,
        **kwargs: Any,
    ) -> None:
        self.name = name

        self._rmm = rmm
        self._batch_size = batch_size
        self._reward_discount = reward_discount

        self._agent2policy: Dict[Any, str] = {}
        self._policies: List[BaseRLPolicy] = []

    @property
    def num_agents(self) -> int:
        return len(self._agent2policy)

    def register_logger(self, logger=None) -> None:  # TODO: typehint
        self._logger = logger

    def register_agent2policy(self, agent2policy: Dict[Any, str], policy2trainer: Dict[str, str]) -> None:
        self._agent2policy = {
            agent_name: policy_name
            for agent_name, policy_name in agent2policy.items()
            if policy2trainer[policy_name] == self.name
        }

    def register_policies(self, policies: List[BaseRLPolicy], policy2trainer: Dict[str, str]) -> None:
        self._policies = [policy for policy in policies if policy2trainer[policy.name] == self.name]

    @abstractmethod
    def train_step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def record_exp(self, env_id: int, exps: List[ExpElement]) -> None:
        raise NotImplementedError

    @abstractmethod
    def build(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_policy_state(self) -> Dict[str, dict]:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def to_device(self, device: torch.device) -> None:
        raise NotImplementedError


class SingleAgentTrainer(BaseTrainer, metaclass=ABCMeta):
    @property
    def policy(self) -> BaseRLPolicy:
        return self._policies[0]

    def register_policies(self, policies: List[BaseRLPolicy], policy2trainer: Dict[str, str]) -> None:
        super().register_policies(policies, policy2trainer)
        assert len(self._policies) == 1

    def record_exp(self, env_id: int, exps: List[ExpElement]) -> None:
        agent_exps_dict = defaultdict(lambda: defaultdict(list))
        for element in exps:
            for agent_name in element.agent_names:
                agent_exps_dict[agent_name]["obs"].append(element.agent_obs_dict[agent_name])
                agent_exps_dict[agent_name]["action"].append(element.policy_action_dict[agent_name])
                agent_exps_dict[agent_name]["reward"].append(element.reward_dict[agent_name])
                agent_exps_dict[agent_name]["terminal"].append(element.terminal_dict[agent_name])
                agent_exps_dict[agent_name]["truncated"].append(element.truncated)
                agent_exps_dict[agent_name]["next_obs"].append(
                    element.next_agent_obs_dict.get(
                        agent_name,
                        element.agent_obs_dict[agent_name],
                    ),
                )

        batches = [Batch(**agent_exps) for agent_exps in agent_exps_dict.values()]
        batch = Batch.cat(batches)
        self._rmm.store(batches=[batch], ids=[env_id])

    @property
    def ops(self) -> BaseTrainOps:
        assert hasattr(self, "_ops")
        return getattr(self, "_ops")

    def get_policy_state(self) -> Dict[str, dict]:
        return {self.policy.name: self.ops.get_state()["policy"]}

    def load(self, path: str) -> None:
        auxiliary_state = torch.load(os.path.join(path, f"trainer__{self.name}.ckpt"))
        self.ops.set_state({"auxiliary": auxiliary_state})

    def save(self, path: str) -> None:
        ops_state = self.ops.get_state()
        auxiliary_state = ops_state["auxiliary"]
        torch.save(auxiliary_state, os.path.join(path, f"trainer__{self.name}.ckpt"))


class MultiAgentTrainer(BaseTrainer, metaclass=ABCMeta):
    pass
