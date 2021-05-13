# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from os import getcwd
from typing import Dict, Union

from maro.rl.experience import ExperienceSet
from maro.rl.policy import AbsPolicy
from maro.utils import Logger

ExperienceTrigger = namedtuple("ExperienceTrigger", ["trigger", "warmup"], defaults=[1])


class AbsPolicyManager(ABC):
    def __init__(self, agent2policy: Dict[str, str]):
        self.agent2policy = agent2policy
        self._policy_names = list(agent2policy.values())

    @property
    def names(self):
        return self._policy_names

    @abstractmethod
    def on_experiences(self, exp_by_agent: Dict[str, ExperienceSet]):
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, state_by_agent):
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        raise NotImplementedError


class LocalPolicyManager(AbsPolicyManager):
    def __init__(
        self,
        policy_dict: Dict[str, AbsPolicy],
        agent2policy: Dict[str, str],
        experience_trigger: Union[ExperienceTrigger, dict] = None,
        log_dir: str = getcwd() 
    ):
        super().__init__(agent2policy)
        self._logger = Logger("LOCAL_POLICY_MANAGER", dump_folder=log_dir)
        self.policy_dict = policy_dict
        self._policy = {agent_id: policy_dict[policy_id] for agent_id, policy_id in self.agent2policy.items()}
        self._agent_groups_by_policy = defaultdict(list)
        for agent_id, policy_id in self.agent2policy.items():
            self._agent_groups_by_policy[policy_id].append(agent_id)

        if experience_trigger is None:
            self._experience_trigger = {}
        elif isinstance(experience_trigger, dict):
            self._experience_trigger = experience_trigger
        else:
            self._experience_trigger = {
                policy_id: experience_trigger for policy_id, policy in self.policy_dict.items()
                if hasattr(policy, "experience_manager") and hasattr(policy, "update")
            }

        self._new_exp_counter = defaultdict(int)
        self._updated_policy_ids = set()

    def choose_action(self, state_by_agent: dict):
        return {agent_id: self._policy[agent_id].choose_action(state) for agent_id, state in state_by_agent.items()}

    def on_experiences(self, exp_by_agent: Dict[str, ExperienceSet]):
        for agent_id, exp in exp_by_agent.items():
            policy_id = self.agent2policy[agent_id]
            policy = self.policy_dict[policy_id]
            if hasattr(policy, "experience_manager"):
                self._new_exp_counter[policy_id] += exp.size
                policy.experience_manager.put(exp)

        for policy_id, policy in self.policy_dict.items():
            if hasattr(policy, "experience_manager"):
                print(f"Policy {policy_id}: exp mem size = {policy.experience_manager.size}, new exp = {self._new_exp_counter[policy_id]}")
                if (
                    policy_id not in self._experience_trigger or
                    policy.experience_manager.size >= self._experience_trigger[policy_id].warmup and
                    self._new_exp_counter[policy_id] >= self._experience_trigger[policy_id].trigger
                ):
                    policy.update()
                    self._new_exp_counter[policy_id] = 0
                    self._updated_policy_ids.add(policy_id)

        if self._updated_policy_ids:
            self._logger.info(f"Updated policies {self._updated_policy_ids}")

    def get_state(self):
        policy_state_dict = {
            policy_id: self.policy_dict[policy_id].get_state() for policy_id in self._updated_policy_ids
        }
        self._updated_policy_ids.clear()
        return policy_state_dict
