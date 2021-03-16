# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import List, Union


class MultiAgentWrapper:
    """Multi-agent wrapper class that exposes the same interfaces as a single agent."""
    def __init__(self, agent_dict: dict):
        self.agent_dict = agent_dict

    def __getitem__(self, agent_id: str):
        return self.agent_dict[agent_id]

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

    def load_model(self, model_dict: dict):
        """Load models from memory for each agent."""
        for agent_id, model in model_dict.items():
            self.agent_dict[agent_id].load_model(model)

    def dump_model(self, agent_ids: Union[str, List[str]] = None):
        """Get agents' underlying models.

        This is usually used in distributed mode where models need to be broadcast to remote roll-out actors.
        """
        if agent_ids is None:
            return {agent_id: agent.dump_model() for agent_id, agent in self.agent_dict.items()}
        elif isinstance(agent_ids, str):
            return self.agent_dict[agent_ids].dump_model()
        else:
            return {agent_id: self.agent_dict[agent_id].dump_model() for agent_id in self.agent_dict}

    def load_model_from_file(self, dir_path):
        """Load models from disk for each agent."""
        for agent_id, agent in self.agent_dict.items():
            agent.load_model_from_file(os.path.join(dir_path, agent_id))

    def dump_model_to_file(self, dir_path: str, agent_ids: Union[str, List[str]] = None):
        """Dump agents' models to disk.

        Each agent will use its own name to create a separate file under ``dir_path`` for dumping.
        """
        os.makedirs(dir_path, exist_ok=True)
        if agent_ids is None:
            for agent_id, agent in self.agent_dict.items():
                agent.dump_model_to_file(os.path.join(dir_path, agent_id))
        elif isinstance(agent_ids, str):
            self.agent_dict[agent_ids].dump_model_to_file(os.path.join(dir_path, agent_ids))
        else:
            for agent_id in agent_ids:
                self.agent_dict[agent_id].dump_model_to_file(os.path.join(dir_path, agent_id))
