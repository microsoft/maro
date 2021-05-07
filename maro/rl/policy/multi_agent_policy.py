# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
import warnings
from collections import defaultdict, namedtuple
from typing import Dict, List

from maro.rl.exploration import AbsExploration, NullExploration
from maro.rl.experience import ExperienceMemory

from .policy import AbsFixedPolicy, AbsCorePolicy


class MultiAgentPolicy:
    """Convenience wrapper of a set of agents that exposes similar interfaces as a single agent.

    Args:
        
    """
    def __init__(
        self,
        policy_dict: Dict[str, AbsPolicy],
        agent_to_policy: Dict[str, str],
        exploration_dict: Dict[str, AbsExploration] = None,
        agent_to_exploration: Dict[str, str] = None
    ):
        self.policy_dict = policy_dict
        self.agent_to_policy = agent_to_policy
        self.policy = {agent_id: policy_dict[policy_id] for agent_id, policy_id in self.agent_to_policy.items()}
        self.agent_groups_by_policy = defaultdict(list)
        for agent_id, policy_id in agent_to_policy.items():
            self.agent_groups_by_policy[policy_id].append(agent_id)

        for policy_id, agent_ids in self.agent_groups_by_policy.items():
            self.agent_groups_by_policy[policy_id] = tuple(agent_ids)

        self.exploration_dict = exploration_dict
        if exploration_dict:
            self.agent_to_exploration = agent_to_exploration
            self.exploration = {
                agent_id: self.exploration_dict[exploration_id]
                for agent_id, exploration_id in self.agent_to_exploration.items()
            }
            self.exploration_enabled = True
            self.agent_groups_by_exploration = defaultdict(list)
            for agent_id, exploration_id in agent_to_exploration.items():
                self.agent_groups_by_exploration[exploration_id].append(agent_id)

            for exploration_id, agent_ids in self.agent_groups_by_exploration.items():
                self.agent_groups_by_exploration[exploration_id] = tuple(agent_ids)

    def train_mode(self):
        if hasattr(self, "exploration_enabled"):
            self.exploration_enabled = True

    def eval_mode(self):
        if hasattr(self, "exploration_enabled"):
            self.exploration_enabled = False

    def choose_action(self, state_by_agent: dict):
        if self.exploration_dict and self.exploration_enabled:      
            return {
                agent_id:
                    self.exploration[agent_id](self.policy[agent_id].choose_action(state))
                    if agent_id in self.exploration else self.policy[agent_id].choose_action(state)
                for agent_id, state in state_by_agent.items()
            }

        return {agent_id: self.policy[agent_id].choose_action(state) for agent_id, state in state_by_agent.items()}

    def store_experiences(self, experiences_by_agent: dict):
        for agent_id, exp in experiences_by_agent.items():
            if isinstance(self.policy[agent_id], AbsCorePolicy):
                self.policy[agent_id].store_experiences(exp)

    def update(self, policy_ids: List[str] = None):
        if policy_ids is None:
            policy_ids = self.policy_dict.keys()

        for policy_id in policy_ids:
            if self.policy_dict[policy_id].update():
                self.policy_dict[policy_id].post_update()

    def load_state(self, policy_state_dict: dict):
        """Load policies from memory."""
        if not policy_state_dict.keys() <= self.policy_dict.keys():
            raise Exception(f"Expected policies from {list(self.policy_state_dict.keys())}")

        for policy_id, policy_state in policy_state_dict.items():
            self.policy_dict[policy_id].load_state(policy_state)

    def get_state(self, policy_ids: List[str] = None):
        if policy_ids is None:
            policy_ids = self.policy_dict.keys()

        return {policy_id: self.policy_dict[policy_id].get_state() for policy_id in policy_ids}

    def load(self, dir_path: str):
        """Load models from disk."""
        for policy_id, policy in self.policy_dict.items():
            try:
                policy.load(os.path.join(dir_path, policy_id))
            except FileNotFoundError:
                warnings.warn(f"policy {policy_id} is skipped because no file is found")

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        for policy_id, policy in self.policy_dict.items():
            policy.save(os.path.join(dir_path, policy_id))
