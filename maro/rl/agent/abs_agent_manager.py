# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from abc import ABC, abstractmethod
from enum import Enum

from maro.rl.shaping.action_shaper import ActionShaper
from maro.rl.shaping.experience_shaper import ExperienceShaper
from maro.rl.shaping.state_shaper import StateShaper
from maro.utils.exception.rl_toolkit_exception import WrongAgentManagerModeError


class AgentManagerMode(Enum):
    TRAIN = "train"
    INFERENCE = "inference"
    TRAIN_INFERENCE = "train_inference"


class AbsAgentManager(ABC):
    """Abstract agent manager class.

    The agent manager provides a unified interactive interface with the environment for RL agent(s). From
    the actor’s perspective, it isolates the complex dependencies of the various homogeneous/heterogeneous
    agents, so that the whole agent manager will behave just like a single agent.

    Args:
        name (str): Name of agent manager.
        mode (AgentManagerMode): An ``AgentManagerNode`` enum member that indicates the role of the agent manager
            in the current process.
        agent_dict (dict): A dictionary of agents to be wrapped by the agent manager.
        experience_shaper (ExperienceShaper, optional): It is responsible for processing data in the replay buffer at
            the end of an episode.
        state_shaper (StateShaper, optional): It is responsible for converting the environment observation to model
            input.
        action_shaper (ActionShaper, optional): It is responsible for converting an agent's model output to environment
            executable action. Cannot be None under Inference and TrainInference modes.
        experience_shaper (ExperienceShaper, optional): It is responsible for processing data in the replay buffer at
            the end of an episode.
    """
    def __init__(
        self,
        name: str,
        mode: AgentManagerMode,
        agent_dict: dict,
        state_shaper: StateShaper = None,
        action_shaper: ActionShaper = None,
        experience_shaper: ExperienceShaper = None
    ):
        self._name = name
        self._mode = mode
        self._agent_dict = agent_dict
        self._state_shaper = state_shaper
        self._action_shaper = action_shaper
        self._experience_shaper = experience_shaper

    def __getitem__(self, agent_id):
        if isinstance(self._agent_dict, dict):
            return self._agent_dict[agent_id]

    @property
    def name(self):
        """Agent manager's name."""
        return self._name

    @property
    def state_shaper(self):
        return self._state_shaper

    @property
    def action_shaper(self):
        return self._action_shaper

    @property
    def experience_shaper(self):
        return self._experience_shaper

    @abstractmethod
    def choose_action(self, *args, **kwargs):
        """Generate an environment executable action given the current decision event and snapshot list.
        """
        return NotImplemented

    @abstractmethod
    def on_env_feedback(self, *args, **kwargs):
        """Processing logic after receiving feedback from the environment is implemented here.

        See ``SimpleAgentManager`` for example.
        """
        return NotImplemented

    @abstractmethod
    def post_process(self, *args, **kwargs):
        """Processing logic after an episode is finished.

        These things may involve generating experiences and resetting stateful objects. See ``SimpleAgentManager``
        for example.
        """
        return NotImplemented

    @abstractmethod
    def train(self, *args, **kwargs):
        """Train the agents."""
        return NotImplemented

    def update_exploration_params(self, exploration_params):
        # Per-agent exploration parameters
        if isinstance(exploration_params, dict) and exploration_params.keys() <= self._agent_dict.keys():
            for agent_id, params in exploration_params.items():
                self._agent_dict[agent_id].update(**params)
        # Shared exploration parameters for all agents
        else:
            for agent in self._agent_dict.values():
                agent.update(**exploration_params)

    def load_models(self, agent_model_dict):
        """Load models from memory for each agent."""
        for agent_id, models in agent_model_dict.items():
            self._agent_dict[agent_id].load_models(models)

    def dump_models(self) -> dict:
        """Get agents' underlying models.

        This is usually used in distributed mode where models need to be broadcast to remote roll-out actors.
        """
        return {agent_id: agent.dump_model() for agent_id, agent in self._agent_dict.items()}

    def load_models_from_files(self, dir_path):
        """Load models from disk for each agent."""
        for agent in self._agent_dict.values():
            agent.load_models_from_file(dir_path)

    def dump_models_to_files(self, dir_path: str):
        """Dump agents' models to disk.

        Each agent will use its own name to create a separate file under ``dir_path`` for dumping.
        """
        os.makedirs(dir_path, exist_ok=True)
        for agent in self._agent_dict.values():
            agent.dump_model_to_file(dir_path)

    def _assert_train_mode(self):
        if self._mode != AgentManagerMode.TRAIN and self._mode != AgentManagerMode.TRAIN_INFERENCE:
            raise WrongAgentManagerModeError(msg=f"this method is unavailable under mode {self._mode}")

    def _assert_inference_mode(self):
        if self._mode != AgentManagerMode.INFERENCE and self._mode != AgentManagerMode.TRAIN_INFERENCE:
            raise WrongAgentManagerModeError(msg=f"this method is unavailable under mode {self._mode}")
