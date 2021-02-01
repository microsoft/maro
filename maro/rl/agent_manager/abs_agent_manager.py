# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from abc import ABC, abstractmethod
from typing import Dict, Union

from maro.communication import Proxy
from maro.rl.agent import AbsAgent
from maro.rl.shaping import Shaper


class AbsAgentManager(ABC):
    """Abstract agent manager class.

    The agent manager provides a unified interactive interface with the environment for RL agent(s). From
    the actor’s perspective, it isolates the complex dependencies of the various homogeneous/heterogeneous
    agents, so that the whole agent manager will behave just like a single agent.

    Args:
        agent (Union[AbsAgent, Dict[str, AbsAgent], Proxy]): Agent or ditionary of agents managed by the agent.
            It may also be a ``Proxy`` instance under distributed mode with a central inference learner.
            See ``AgentManagerProxy`` for details.
        state_shaper (Shaper, optional): It is responsible for converting the environment observation to model
            input.
        action_shaper (Shaper, optional): It is responsible for converting an agent's model output to environment
            executable action. Cannot be None under Inference and TrainInference modes.
        experience_shaper (Shaper, optional): It is responsible for processing data in the replay buffer at
            the end of an episode.
    """
    def __init__(
        self,
        agent: Union[AbsAgent, Dict[str, AbsAgent], Proxy],
        state_shaper: Shaper = None,
        action_shaper: Shaper = None,
        experience_shaper: Shaper = None
    ):
        self.agent = agent
        self._state_shaper = state_shaper
        self._action_shaper = action_shaper
        self._experience_shaper = experience_shaper

    @abstractmethod
    def choose_action(self, *args, **kwargs):
        """Generate an environment executable action given the current decision event and snapshot list.
        """
        return NotImplemented

    @abstractmethod
    def on_env_feedback(self, *args, **kwargs):
        """Processing logic after receiving feedback from the environment is implemented here.

        See ``AgentManager`` for example.
        """
        return NotImplemented

    @abstractmethod
    def post_process(self, *args, **kwargs):
        """Processing logic after an episode is finished.

        These things may involve generating experiences and resetting stateful objects. See ``AgentManager``
        for example.
        """
        return NotImplemented

    @abstractmethod
    def train(self, *args, **kwargs):
        """Train agents."""
        return NotImplemented

    def set_exploration_params(self, params):
        # Per-agent exploration parameters
        if isinstance(self.agent, AbsAgent):
            self.agent.set_exploration_params(**params)
        elif isinstance(self.agent, dict):
            if isinstance(params, dict) and params.keys() <= self.agent.keys():
                for agent_id, params in params.items():
                    self.agent[agent_id].set_exploration_params(**params)
            # Shared exploration parameters for all agents
            else:
                for agent in self.agent.values():
                    agent.set_exploration_params(**params)

    def load_models(self, agent_model_dict):
        """Load models from memory for each agent."""
        if isinstance(self.agent, AbsAgent):
            self.agent.load_model()
        elif isinstance(self.agent, dict):
            for agent_id, models in agent_model_dict.items():
                self.agent[agent_id].load_model(models)

    def dump_models(self) -> dict:
        """Get agents' underlying models.

        This is usually used in distributed mode where models need to be broadcast to remote roll-out actors.
        """
        if isinstance(self.agent, AbsAgent):
            return self.agent.dump_model()
        elif isinstance(self.agent, dict):
            return {agent_id: agent.dump_model() for agent_id, agent in self.agent.items()}

    def load_models_from_files(self, dir_path):
        """Load models from disk for each agent."""
        if isinstance(self.agent, AbsAgent):
            self.agent.load_model_from_file(dir_path)
        elif isinstance(self.agent, dict):
            for agent in self.agent.values():
                agent.load_model_from_file(dir_path)

    def dump_models_to_files(self, dir_path: str):
        """Dump agents' models to disk.

        Each agent will use its own name to create a separate file under ``dir_path`` for dumping.
        """
        os.makedirs(dir_path, exist_ok=True)
        if isinstance(self.agent, AbsAgent):
            self.agent.dump_model_to_file(dir_path)
        elif isinstance(self.agent, dict):
            for agent in self.agent.values():
                agent.dump_model_to_file(dir_path)
