# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from abc import ABC, abstractmethod
from enum import Enum

from maro.rl.explorer.abs_explorer import AbsExplorer
from maro.rl.shaping.action_shaper import ActionShaper
from maro.rl.shaping.experience_shaper import ExperienceShaper
from maro.rl.shaping.state_shaper import StateShaper
from maro.utils.exception.rl_toolkit_exception import MissingShaperError, UnsupportedAgentModeError, WrongAgentModeError


class AgentMode(Enum):
    TRAIN = "train"
    INFERENCE = "inference"
    TRAIN_INFERENCE = "train_inference"


class AbsAgentManager(ABC):
    """Abstract agent manager class.

    The agent manager provides a unified interactive interface with the environment for RL agent(s). From
    the actor’s perspective, it isolates the complex dependencies of the various homogeneous/heterogeneous
    agents, so that the whole agent manager will behave just like a single agent. Besides that, the agent
    manager also plays the role of an agent assembler. It can assemble different RL agents according to the
    actual requirements, such as whether to share the underlying model, whether to share the experience
    pool, etc.

    Args:
        name (str): Name of agent manager.
        mode (AgentMode): An AgentMode enum member that specifies that role of the agent. Some attributes may
            be None under certain modes.
        agent_id_list (list): List of agent identifiers.
        experience_shaper (ExperienceShaper, optional): It is responsible for processing data in the replay buffer at
            the end of an episode.
        state_shaper (StateShaper, optional): It is responsible for converting the environment observation to model
            input.
        action_shaper (ActionShaper, optional): It is responsible for converting an agent's model output to environment
            executable action. Cannot be None under Inference and TrainInference modes.
        explorer (AbsExplorer): It is responsible for storing and updating exploration rates.
    """
    def __init__(
        self, name: str, mode: AgentMode, agent_id_list: [str], state_shaper: StateShaper = None,
        action_shaper: ActionShaper = None, experience_shaper: ExperienceShaper = None, explorer: AbsExplorer = None
    ):
        self._name = name
        if mode not in AgentMode:
            raise UnsupportedAgentModeError(msg='mode must be "train", "inference" or "train_inference"')
        self._mode = mode

        if mode in {AgentMode.INFERENCE, AgentMode.TRAIN_INFERENCE}:
            if state_shaper is None:
                raise MissingShaperError(msg=f"state shaper cannot be None under mode {self._mode}")
            if action_shaper is None:
                raise MissingShaperError(msg=f"action_shaper cannot be None under mode {self._mode}")
            if experience_shaper is None:
                raise MissingShaperError(msg=f"experience_shaper cannot be None under mode {self._mode}")

        self._state_shaper = state_shaper
        self._action_shaper = action_shaper
        self._experience_shaper = experience_shaper
        self._explorer = explorer

        self._agent_id_list = agent_id_list
        self._trajectory = []

        self._agent_dict = {}
        self._assemble(self._agent_dict)

    def __getitem__(self, agent_id):
        return self._agent_dict[agent_id]

    def _assemble(self, agent_dict):
        """Assembles agents and fill the ``agent_dict`` with them."""
        return NotImplemented

    def choose_action(self, decision_event, snapshot_list):
        """This is the interface for interacting with the environment.

        The method consists of 4 steps:

        1. The decision event and snapshot list are converted by the state shaper to a model input. \
            The state shaper also finds the target agent ID.
        2. The target agent takes the model input and uses its underlying models to compute an action.
        3. Key information regarding the transition is recorded in the ``_trajectory`` attribute.
        4. The action computed by the model is converted to an environment executable action by the action shaper.

        Args:
            decision_event: A decision event that prompts an action.
            snapshot_list: An object that holds the detailed history of past env observations.

        Returns:
            An action object that can be passed directly to an environment's ``step`` method.
        """
        self._assert_inference_mode()
        agent_id, model_state = self._state_shaper(decision_event, snapshot_list)
        model_action = self._agent_dict[agent_id].choose_action(
            model_state, self._explorer.epsilon[agent_id] if self._explorer else None
        )
        self._trajectory.append(
            {
                "state": model_state,
                "action": model_action,
                "reward": None,
                "agent_id": agent_id,
                "event": decision_event
            }
        )
        return self._action_shaper(model_action, decision_event, snapshot_list)

    def on_env_feedback(self, metrics):
        """This method records the environment-generated metrics as part of the latest transition in the trajectory.

        Args:
            metrics: business metrics provided by the environment after an action has been executed.
        """
        self._trajectory[-1]["metrics"] = metrics

    def post_process(self, snapshot_list):
        """This method processes the latest trajectory into experiences.

        Args:
            snapshot_list: the snapshot list from the env at the end of an episode.
        """
        experiences = self._experience_shaper(self._trajectory, snapshot_list)
        self._trajectory.clear()
        self._state_shaper.reset()
        self._action_shaper.reset()
        self._experience_shaper.reset()
        return experiences

    @abstractmethod
    def store_experiences(self, experiences):
        """Abstract method to store experiences generated by the experience shaper in the experience pool(s).

        Depending on the user's implementation of the experience shaper's ``__call__`` method, ``experiences`` may
        come in different formats (e.g., a dictionary with agent ID's as keys). The user must implement this
        method in accordance with this format.

        Args:
            experiences: experiences generated by the experience shaper during post-processing.
        """
        return NotImplementedError

    def update_epsilon(self, performance):
        """This method updates the exploration rates for each agent.

        Args:
            performance: Performance from the latest episode. Depending on the implementation of the explorer,
                this may or may not be used in generating the exploration rates.
        """
        if self._explorer:
            self._explorer.update(performance)

    def train(self):
        """Train all agents."""
        self._assert_train_mode()
        for agent in self._agent_dict.values():
            agent.train()

    def load_models(self, agent_model_dict):
        """Load models from memory for each agent."""
        for agent_id, model_dict in agent_model_dict.items():
            self._agent_dict[agent_id].load_model_dict(model_dict)

    def load_models_from_files(self, file_path_dict):
        """Load models from disk for each agent."""
        for agent_id, file_path in file_path_dict.items():
            self._agent_dict[agent_id].load_model_dict_from(file_path)

    def dump_models(self, dir_path: str):
        """Dump agents' models to disk.

        Each agent will use its own name to create a separate file under ``dir_path`` for dumping.
        """
        os.makedirs(dir_path, exist_ok=True)
        for agent in self._agent_dict.values():
            agent.dump_model_dict(dir_path)

    def get_models(self):
        """Get agents' underlying models.

        This is usually used in distributed mode where models need to be broadcast to remote roll-out actors.
        """
        return {agent_id: agent.algorithm.model_dict for agent_id, agent in self._agent_dict.items()}

    @property
    def name(self):
        """Agent manager's name."""
        return self._name

    @property
    def agents(self):
        """Agents managed by the agent manager."""
        return self._agent_dict

    @property
    def explorer(self):
        """Explorer used by the agent manager."""
        return self._explorer

    def _assert_train_mode(self):
        if self._mode != AgentMode.TRAIN and self._mode != AgentMode.TRAIN_INFERENCE:
            raise WrongAgentModeError(msg=f"this method is unavailable under mode {self._mode}")

    def _assert_inference_mode(self):
        if self._mode != AgentMode.INFERENCE and self._mode != AgentMode.TRAIN_INFERENCE:
            raise WrongAgentModeError(msg=f"this method is unavailable under mode {self._mode}")
