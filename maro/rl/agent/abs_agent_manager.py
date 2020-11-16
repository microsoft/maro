# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterator, Union

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
        agent_dict (dict): A dictionary of agents to be wrapper by the agent manager.
        experience_shaper (ExperienceShaper, optional): It is responsible for processing data in the replay buffer at
            the end of an episode.
        state_shaper (StateShaper, optional): It is responsible for converting the environment observation to model
            input.
        action_shaper (ActionShaper, optional): It is responsible for converting an agent's model output to environment
            executable action. Cannot be None under Inference and TrainInference modes.
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
        self.agent_dict = agent_dict
        self._state_shaper = state_shaper
        self._action_shaper = action_shaper
        self._experience_shaper = experience_shaper

    def __getitem__(self, agent_id):
        return self.agent_dict[agent_id]

    @property
    def name(self):
        """Agent manager's name."""
        return self._name

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

    def register_exploration_schedule(self, exploration_schedule: Union[Iterator, dict]):
        for agent_id, agent in self.agent_dict.items():
            agent.explorer.register_schedule(
                exploration_schedule[agent_id] if isinstance(exploration_schedule, dict) else exploration_schedule
            )

    def load_exploration_params(self, exploration_params):
        is_per_agent = set(exploration_params).issubset(set(self.agent_dict.keys()))
        if is_per_agent:
            for agent_id, params in exploration_params.items():
                self.agent_dict[agent_id].load_exploration_params(params)
        else:
            for agent in self.agent_dict.values():
                agent.load_exploration_params(exploration_params)

    def dump_exploration_params(self):
        return {agent_id: agent.dump_exploration_params() for agent_id, agent in self.agent_dict.items()}

    def update_exploration_params(self):
        for agent in self.agent_dict.values():
            agent.explorer.update()

    def _assert_train_mode(self):
        if self._mode != AgentManagerMode.TRAIN and self._mode != AgentManagerMode.TRAIN_INFERENCE:
            raise WrongAgentManagerModeError(msg=f"this method is unavailable under mode {self._mode}")

    def _assert_inference_mode(self):
        if self._mode != AgentManagerMode.INFERENCE and self._mode != AgentManagerMode.TRAIN_INFERENCE:
            raise WrongAgentManagerModeError(msg=f"this method is unavailable under mode {self._mode}")
