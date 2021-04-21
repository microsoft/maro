# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
from typing import Dict, Union

from maro.rl.algorithm import AbsFixedPolicy, AbsTrainablePolicy
from maro.rl.exploration import AbsExplorer
from maro.rl.storage import SimpleStore


class TrainingConfig:
    __slots__ = ["sampler_cls", "sampler_params", "train_iters", "min_experiences_to_trigger_learning"]

    def __init__(
        self,
        sampler_cls,
        sampler_params: dict,
        train_iters: int,
        min_experience_to_trigger_learning: int = 1
    ):
        self.sampler_cls = sampler_cls
        self.sampler_params = sampler_params
        self.train_iters = train_iters
        self.min_experiences_to_trigger_learning = min_experiences_to_trigger_learning


class MultiAgentPolicyForInference:
    """Convenience wrapper of a set of agents that exposes similar interfaces as a single agent.

    Args:
        
    """
    def __init__(
        self,
        policy_dict: Dict[str, Union[AbsFixedPolicy, AbsTrainablePolicy]],
        agent_to_policy: Dict[str, str],
        explorer_dict: Dict[str, AbsExplorer] = None,
        agent_to_explorer: Dict[str, str] = None
    ):
        self.policy_dict = policy_dict
        self.policy = {agent_id: policy_dict[policy_id] for agent_id, policy_id in agent_to_policy.items()}

        if explorer_dict is not None:
            self.explorer = {
                agent_id: explorer_dict[explorer_id] for agent_id, explorer_id in agent_to_explorer.items()
            }

    def choose_action(self, state_by_agent: dict):
        return {agent_id: self.policy[agent_id].choose_action(state) for agent_id, state in state_by_agent.items()}

    def update_exploration_params(self, param_dict: dict):
        # Per-agent exploration parameters
        for policy_id, params in param_dict.items():
            self.policy_dict[policy_id].set_exploration_params(params)

    def load(self, policy_dict: dict):
        """Load models from memory."""
        self.policy_dict = policy_dict
        self.policy = {agent_id: policy_dict[policy_id] for agent_id, policy_id in agent_to_policy.items()}

    def load_from_file(self, path: str):
        """Load models from disk."""
        with open(path, "rb") as fp:
            pickle.load(fp)


class MultiAgentPolicyForTraining:
    """Convenience wrapper of a set of agents that exposes similar interfaces as a single agent.

    Args:
        
    """
    def __init__(
        self,
        policy_dict: Dict[str, Union[AbsFixedPolicy, AbsTrainablePolicy]],
        agent_to_policy: Dict[str, str],
        experience_memory_dict: Dict[str, SimpleStore],
        agent_to_experience_memory: Dict[str, str],
        training_config_dict: Dict[str, TrainingConfig],
        agent_to_training_config: Dict[str, str]
    ):
        self.policy_dict = policy_dict
        self.policy = {agent_id: policy_dict[policy_id] for agent_id, policy_id in agent_to_policy.items()}

        assert agent_to_experience_memory.keys() == agent_to_training_config.keys()
        self.experience_memory = {
            agent_id: experience_memory_dict[mem_id] for agent_id, mem_id in agent_to_experience_memory.items()
        }

        self.training_config = {
            agent_id: training_config_dict[config_id] for agent_id, config_id in agent_to_training_config.items()
        }

        self.sampler = {
            agent_id: cfg.sampler_cls(self.experience_memory[agent_id], **cfg.sampler_params)
            for agent_id, cfg in self.training_config.items()
        }

    def store_experiences(self, experiences_by_agent: dict):
        """Store experiences in the agents' experience memory.

        The top-level keys of ``experiences`` will be treated as agent IDs.
        """
        for agent_id, exp in experiences_by_agent.items():
            self.experience_memory[agent_id].put(exp)

    def update(self):
        for agent_id, config in self.training_config.items(): 
            if len(self.experience_memory[agent_id]) >= config.min_experiences_to_trigger_learning:
                for _ in config.train_iters:
                    self.policy[agent_id].update(self.sampler[agent_id].sample())

    def save(self, path: dir):
        with open(path, "wb") as fp:
            pickle.save(self.policy_dict, path)
