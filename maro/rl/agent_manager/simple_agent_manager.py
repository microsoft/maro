# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from abc import abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Dict

from maro.rl.agent import AbsAgent, ActionInfo
from maro.rl.shaping import Shaper
from maro.utils.exception.rl_toolkit_exception import AgentManagerModeError, MissingShaper

from .abs_agent_manager import AbsAgentManager


class AgentManagerMode(Enum):
    TRAIN = "train"
    INFERENCE = "inference"
    TRAIN_INFERENCE = "train_inference"


class AgentManager(AbsAgentManager):
    """Abstract agent manager class.

    The agent manager provides a unified interactive interface with the environment for RL agent(s). From
    the actor’s perspective, it isolates the complex dependencies of the various homogeneous/heterogeneous
    agents, so that the whole agent manager will behave just like a single agent.

    Args:
        name (str): Name of agent manager.
        mode (AgentManagerMode): An ``AgentManagerNode`` enum member that indicates the role of the agent manager
            in the current process.
        agents (Dict[str, AbsAgent]): A dictionary of agents to be wrapper by the agent manager.
        state_shaper (Shaper, optional): It is responsible for converting the environment observation to model
            input.
        action_shaper (Shaper, optional): It is responsible for converting an agent's model output to environment
            executable action. Cannot be None under Inference and TrainInference modes.
        experience_shaper (Shaper, optional): It is responsible for processing data in the replay buffer at
            the end of an episode.
    """
    def __init__(
        self,
        name: str,
        mode: AgentManagerMode,
        agents: Dict[str, AbsAgent],
        state_shaper: Shaper = None,
        action_shaper: Shaper = None,
        experience_shaper: Shaper = None
    ):
        if mode in {AgentManagerMode.INFERENCE, AgentManagerMode.TRAIN_INFERENCE}:
            if state_shaper is None:
                raise MissingShaper(msg=f"state shaper cannot be None under mode {self._mode}")
            if action_shaper is None:
                raise MissingShaper(msg=f"action_shaper cannot be None under mode {self._mode}")
            if experience_shaper is None:
                raise MissingShaper(msg=f"experience_shaper cannot be None under mode {self._mode}")

        super().__init__(
            agents,
            state_shaper=state_shaper,
            action_shaper=action_shaper,
            experience_shaper=experience_shaper
        )
        self._name = name
        self._mode = mode
        # Data structure to temporarily store the trajectory
        self._trajectory = defaultdict(list)

    def __getitem__(self, agent_id):
        return self.agents[agent_id]

    @property
    def name(self):
        """Agent manager's name."""
        return self._name

    def choose_action(self, decision_event, snapshot_list):
        self._assert_inference_mode()
        agent_id, model_state = self._state_shaper(decision_event, snapshot_list)
        action_info = self.agents[agent_id].choose_action(model_state)
        self._trajectory["state"].append(model_state)
        self._trajectory["agent_id"].append(agent_id)
        self._trajectory["event"].append(decision_event)
        if isinstance(action_info, ActionInfo):
            self._trajectory["action"].append(action_info.action)
            self._trajectory["log_action_probability"].append(action_info.log_prob)
        else:
            self._trajectory["action"].append(action_info)

        return self._action_shaper(self._trajectory["action"][-1], decision_event)

    def on_env_feedback(self, metrics):
        """This method records the environment-generated metrics as part of the latest transition in the trajectory.

        Args:
            metrics: business metrics provided by the environment after an action has been executed.
        """
        self._trajectory["metrics"].append(metrics)

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
    def train(self, experiences_by_agent: dict):
        """Train all agents."""
        return NotImplementedError

    def set_exploration_params(self, params):
        # Per-agent exploration parameters
        if isinstance(params, dict) and params.keys() <= self.agents.keys():
            for agent_id, params in params.items():
                self.agents[agent_id].set_exploration_params(**params)
        # Shared exploration parameters for all agents
        else:
            for agent in self.agents.values():
                agent.set_exploration_params(**params)

    def load_models(self, agent_model_dict):
        """Load models from memory for each agent."""
        for agent_id, models in agent_model_dict.items():
            self.agents[agent_id].load_model(models)

    def dump_models(self) -> dict:
        """Get agents' underlying models.

        This is usually used in distributed mode where models need to be broadcast to remote roll-out actors.
        """
        return {agent_id: agent.dump_model() for agent_id, agent in self.agents.items()}

    def load_models_from_files(self, dir_path):
        """Load models from disk for each agent."""
        for agent in self.agents.values():
            agent.load_model_from_file(dir_path)

    def dump_models_to_files(self, dir_path: str):
        """Dump agents' models to disk.

        Each agent will use its own name to create a separate file under ``dir_path`` for dumping.
        """
        os.makedirs(dir_path, exist_ok=True)
        for agent in self.agents.values():
            agent.dump_model_to_file(dir_path)

    def _assert_train_mode(self):
        if self._mode != AgentManagerMode.TRAIN and self._mode != AgentManagerMode.TRAIN_INFERENCE:
            raise AgentManagerModeError(msg=f"this method is unavailable under mode {self._mode}")

    def _assert_inference_mode(self):
        if self._mode != AgentManagerMode.INFERENCE and self._mode != AgentManagerMode.TRAIN_INFERENCE:
            raise AgentManagerModeError(msg=f"this method is unavailable under mode {self._mode}")
