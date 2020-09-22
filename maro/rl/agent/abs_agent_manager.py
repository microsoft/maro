# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from enum import Enum
import os

from maro.rl.shaping.state_shaper import StateShaper
from maro.rl.shaping.action_shaper import ActionShaper
from maro.rl.shaping.experience_shaper import ExperienceShaper
from maro.rl.explorer.abs_explorer import AbsExplorer
from maro.utils.exception.rl_toolkit_exception import UnsupportedAgentModeError, MissingShaperError, WrongAgentModeError


class AgentMode(Enum):
    TRAIN = "train"
    INFERENCE = "inference"
    TRAIN_INFERENCE = "train_inference"


class AbsAgentManager(ABC):
    def __init__(self,
                 name: str,
                 mode: AgentMode,
                 agent_id_list: [str],
                 state_shaper: StateShaper = None,
                 action_shaper: ActionShaper = None,
                 experience_shaper: ExperienceShaper = None,
                 explorer: AbsExplorer = None):
        """
        Manages all agents.

        Args:
            name (str): name of agent manager.
            mode (AgentMode): An AgentMode enum member that specifies that role of the agent. Some attributes may
                              be None under certain modes.
            agent_id_list (list): list of agent identifiers.
            experience_shaper: responsible for processing data in the replay buffer at the end of an episode, e.g.,
                               adjusting rewards and computing target states.
            state_shaper:  responsible for extracting information from a decision event and the snapshot list for
                           the event to form a state vector as accepted by the underlying algorithm
            action_shaper: responsible for converting the output of an agent's action to an EnvAction object that can
                           be executed by the environment. Cannot be None under Inference and TrainInference modes.
            explorer: responsible for storing and updating exploration rates.
        """
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
        """
        abstract method to populate the _agent_dict attribute.
        """
        return NotImplemented

    def choose_action(self, decision_event, snapshot_list):
        self._assert_inference_mode()
        agent_id, model_state = self._state_shaper(decision_event, snapshot_list)
        model_action = self._agent_dict[agent_id].choose_action(
            model_state, self._explorer.epsilon[agent_id] if self._explorer else None)
        self._trajectory.append({"state": model_state,
                                 "action": model_action,
                                 "reward": None,
                                 "agent_id": agent_id,
                                 "event": decision_event})
        return self._action_shaper(model_action, decision_event, snapshot_list)

    def on_env_feedback(self, metrics):
        self._trajectory[-1]["metrics"] = metrics

    def post_process(self, snapshot_list):
        """
        Called at the end of an episode, this function processes data from the latest episode, including reward
        adjustments and next-state computations, and returns experiences for individual agents.

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
        return NotImplementedError

    def update_epsilon(self, performance):
        """
        This updates the exploration rates for each agent.

        Args:
            performance: performance from the latest episode.
        """
        if self._explorer:
            self._explorer.update(performance)

    def train(self):
        self._assert_train_mode()
        for agent in self._agent_dict.values():
            agent.train()

    def load_models(self, agent_model_dict):
        for agent_id, model_dict in agent_model_dict.items():
            self._agent_dict[agent_id].load_model_dict(model_dict)

    def load_models_from_files(self, file_path_dict):
        for agent_id, file_path in file_path_dict.items():
            self._agent_dict[agent_id].load_model_dict_from(file_path)

    def dump_models(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        for agent in self._agent_dict.values():
            agent.dump_model_dict(dir_path)

    def get_models(self):
        return {agent_id: agent.algorithm.model_dict for agent_id, agent in self._agent_dict.items()}

    @property
    def name(self):
        return self._name

    @property
    def agents(self):
        return self._agent_dict

    @property
    def explorer(self):
        return self._explorer

    def _assert_train_mode(self):
        if self._mode != AgentMode.TRAIN and self._mode != AgentMode.TRAIN_INFERENCE:
            raise WrongAgentModeError(msg=f"this method is unavailable under mode {self._mode}")

    def _assert_inference_mode(self):
        if self._mode != AgentMode.INFERENCE and self._mode != AgentMode.TRAIN_INFERENCE:
            raise WrongAgentModeError(msg=f"this method is unavailable under mode {self._mode}")
