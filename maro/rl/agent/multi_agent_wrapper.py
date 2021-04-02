# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import List, Union

from .abs_agent import AbsAgent


class MultiAgentWrapper:
    """Convenience wrapper of a set of agents that exposes similar interfaces as a single agent.
    
    Args:
        agent_dict (Union[AbsAgent, dict]): A single agent or a homogeneous set of agents that have the same
            method signatures.
    """
    def __init__(self, agent_dict: Union[AbsAgent, dict]):
        if isinstance(agent_dict, AbsAgent):
            agent_dict = {"AGENT": agent_dict}
        self.agent_dict = agent_dict
        self._names = list(self.agent_dict.keys())

    def __getitem__(self, agent_id):
        if len(self.agent_dict) == 1:
            return self.agent_dict["AGENT"]
        else:
            return self.agent_dict[agent_id]

    def __len__(self):
        return len(self.agent_dict)

    @property
    def names(self):
        return self._names

    def choose_action(self, state_by_agent: dict):
        return {agent_id: self.agent_dict[agent_id].choose_action(state) for agent_id, state in state_by_agent.items()}

    def set_exploration_params(self, params):
        # Per-agent exploration parameters
        if isinstance(params, dict) and params.keys() <= self.agent_dict.keys():
            for agent_id, params in params.items():
                self.agent_dict[agent_id].set_exploration_params(**params)
        # Shared exploration parameters for all agents
        else:
            for agent in self.agent_dict.values():
                agent.set_exploration_params(**params)

    def store_experiences(self, experiences: dict):
        """Store experiences in the agents' experience memory.
        
        The top-level keys of ``experiences`` will be treated as agent IDs. 
        """
        for agent_id, exp in experiences.items():
            self.agent_dict[agent_id].store_experiences(exp)

    def learn(self, agent_ids=None):
        if agent_ids is None:
            for agent in self.agent_dict.values():
                agent.learn()
        elif not isinstance(agent_ids, list):
            self.agent_dict[agent_ids].learn()
        else:
            for agent_id in agent_ids:
                self.agent_dict[agent_id].learn()

    def load_model(self, model_dict: dict):
        """Load models from memory for each agent."""
        for agent_id, model in model_dict.items():
            self.agent_dict[agent_id].load_model(model)

    def dump_model(self, agent_ids=None):
        """Get agents' underlying models.

        This is usually used in distributed mode where models need to be broadcast to remote roll-out actors.
        """
        if agent_ids is None:
            return {agent_id: agent.dump_model() for agent_id, agent in self.agent_dict.items()}
        elif not isinstance(agent_ids, list):
            return self.agent_dict[agent_ids].dump_model()
        else:
            return {agent_id: self.agent_dict[agent_id].dump_model() for agent_id in self.agent_dict}

    def load_model_from_file(self, dir_path):
        """Load models from disk for each agent."""
        for agent_id, agent in self.agent_dict.items():
            agent.load_model_from_file(os.path.join(dir_path, agent_id))

    def dump_model_to_file(self, dir_path: str, agent_ids=None):
        """Dump agents' models to disk.

        Each agent will use its own name to create a separate file under ``dir_path`` for dumping.
        """
        os.makedirs(dir_path, exist_ok=True)
        if agent_ids is None:
            for agent_id, agent in self.agent_dict.items():
                agent.dump_model_to_file(os.path.join(dir_path, agent_id))
        elif not isinstance(agent_ids, list):
            self.agent_dict[agent_ids].dump_model_to_file(os.path.join(dir_path, agent_ids))
        else:
            for agent_id in agent_ids:
                self.agent_dict[agent_id].dump_model_to_file(os.path.join(dir_path, agent_id))
