# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Dict, Union

from .agent import AbsAgent, AgentGroup


class AgentManager:
    def __init__(self, child_agent_manager_dict: Dict[str, Union[AbsAgent, AgentGroup]]):
        self.child_agent_manager_dict = child_agent_manager_dict

    def choose_action(self, state_dict: dict):
        return {name: self.child_agent_manager_dict[name].choose_action(state) for name, state in state_dict.items()}

    def set_exploration_params(self, param_dict: dict):
        if param_dict.keys() <= self.child_agent_manager_dict.keys():
            for name, param in param_dict.items():
                self.child_agent_manager_dict[name].set_exploration_params(param)
        else:
            for manager in self.child_agent_manager_dict.values():
                if isinstance(manager, AbsAgent):
                    manager.set_exploration_params(**param_dict)
                else:
                    manager.set_exploration_params(param_dict)

    def learn(self, experience_dict: dict) -> set:
        """Store experiences in the agents' experience memory.

        The top-level keys of ``experiences`` will be treated as child agent manager IDs.
        """
        return {name: self.child_agent_manager_dict[name].learn(exp) for name, exp in experience_dict.items()}

    def step(self):
        for manager in self.child_agent_manager_dict.values():
            manager.step()

    def load_model(self, model_dict: dict):
        """Load models from memory."""
        for name, model in model_dict.items():
            self.child_agent_manager_dict[name].load_model(model)

    def dump_model(self):
        """Get agents' underlying models.

        This is usually used in distributed mode where models need to be broadcast to remote roll-out actors.
        """
        return {name: manager.dump_model() for name, manager in self.child_agent_manager_dict.items()}

    def load_model_from_file(self, dir_path):
        """Load models from disk."""
        for name, manager in self.child_agent_manager_dict.items():
            manager.load_model_from_file(os.path.join(dir_path, name))

    def dump_model_to_file(self, dir_path: str):
        """Dump agents' models to disk.

        each agent will use its own name to create a separate file under ``path`` for dumping.
        """
        for name, manager in self.child_agent_manager_dict.items():
            sub_dir = os.path.join(dir_path, name)
            os.makedirs(sub_dir, exist_ok=True)
            manager.dump_model_to_file(sub_dir)
