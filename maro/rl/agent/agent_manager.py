# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from enum import Enum
import os
import pickle

from maro.rl.common import ExperienceKey, ExperienceInfoKey, TransitionInfoKey
from maro.rl.shaping.abstract_state_shaper import AbstractStateShaper
from maro.rl.shaping.abstract_action_shaper import AbstractActionShaper
from maro.rl.shaping.abstract_reward_shaper import AbstractRewardShaper
from maro.rl.explorer.abstract_explorer import AbstractExplorer
from maro.utils import set_seeds


class AgentMode(Enum):
    TRAIN = "train"
    INFERENCE = "inference"
    TRAIN_INFERENCE = "train_inference"


class AgentManager(object):
    def __init__(self,
                 name: str,
                 mode: AgentMode,
                 agent_id_list: [str],
                 state_shaper: AbstractStateShaper = None,
                 action_shaper: AbstractActionShaper = None,
                 reward_shaper: AbstractRewardShaper = None,
                 explorer: AbstractExplorer = None,
                 seed: int = None):
        """
        Manages all agents.
        Args:
            name (str): name of agent manager.
            mode (AgentMode): An AgentMode enum member that specifies that role of the agent. Some attributes may   \n
                              be None under certain modes.
            agent_id_list (list): list of agent identifiers.
            reward_shaper: responsible for processing data in the replay buffer at the end of an episode, e.g.,  \n
                            adjusting rewards and computing target states.
            state_shaper:  responsible for extracting information from a decision event and the snapshot list for  \n
                           the event to form a state vector as accepted by the underlying algorithm
            action_shaper: responsible for converting the output of an agent's action to an EnvAction object that can \n
                           be executed by the environment. Cannot be None under Inference and TrainInference modes.
            explorer: responsible for storing and updating exploration rates.
            seed (int): random seed. If not None, the seeds for numpy, torch and python3's own random module will   \n
                        be set to this value.
        """
        self._name = name
        if mode not in AgentMode:
            raise ValueError('mode must be "train", "inference" or "train_inference"')
        self._mode = mode

        if mode in {AgentMode.INFERENCE, AgentMode.TRAIN_INFERENCE}:
            assert state_shaper is not None, f"state shaper cannot be None under mode {self._mode}"
            assert action_shaper is not None, f"action_shaper cannot be None under mode {self._mode}"
            assert reward_shaper is not None, f"reward_shaper cannot be None under mode {self._mode}"

        self._state_shaper = state_shaper
        self._action_shaper = action_shaper
        self._reward_shaper = reward_shaper
        self._explorer = explorer

        self._current_transition = {}
        self._agent_id_list = agent_id_list
        self._agent_dict = {}

        if seed is not None:
            set_seeds(seed)

        self._assemble_agents()

    @abstractmethod
    def _assemble_agents(self):
        """
        abstract method to populate the _agent_dict attribute.
        """
        return NotImplemented

    def choose_action(self, decision_event, snapshot_list):
        self._assert_inference_mode()
        agent_id, model_state = self._state_shaper(decision_event, snapshot_list)
        model_action = self._agent_dict[agent_id].choose_action(
            model_state, self._explorer.epsilon[agent_id] if self._explorer else None)
        self._current_transition = {ExperienceKey.STATE: model_state,
                                    ExperienceKey.ACTION: model_action,
                                    ExperienceKey.REWARD: None,  # place holder
                                    ExperienceKey.NEXT_STATE: None,  # place holder
                                    ExperienceKey.NEXT_ACTION: None,  # place holder
                                    "extra": {TransitionInfoKey.EVENT: pickle.dumps(decision_event),
                                              TransitionInfoKey.AGENT_ID: agent_id},
                                    "info": {ExperienceInfoKey.TD_ERROR: 1e8,
                                             ExperienceInfoKey.DISCOUNT: None}   # place holder
                                    }
        return self._action_shaper(model_action, decision_event, snapshot_list)

    def on_env_feedback(self, metrics):
        self._current_transition["extra"].update({TransitionInfoKey.METRICS: metrics})
        self._reward_shaper.push(self._current_transition)

    def post_process(self, snapshot_list):
        """
        Called at the end of an episode, this function processes data from the latest episode, including reward  \n
        adjustments and next-state computations, and returns experiences for individual agents.
        Args:
            snapshot_list: the snapshot list from the env at the end of an episode.
        """
        return self._reward_shaper(snapshot_list)

    def store_experiences(self, exp_by_agent):
        for agent_id, exp_dict in exp_by_agent.items():
            self._agent_dict[agent_id].store_experiences(exp_dict)

    def update_epsilon(self, performance):
        """
        Args:
            performance: performance from the latest episode.
        """
        if self._explorer:
            self._explorer.update(performance)

    def train(self):
        for agent in self._agent_dict.values():
            agent.train()

    def load_models(self, model_batch):
        for agent_id, model_dict in model_batch.items():
            self._agent_dict[agent_id].load_model_dict(model_dict)

    def load_models_from(self, path_dict):
        for agent_id, path in path_dict.items():
            self._agent_dict[agent_id].load_model_dict_from(path)

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
        return self._agent_id_list

    @property
    def explorer(self):
        return self._explorer

    def _assert_train_mode(self):
        if self._mode != AgentMode.TRAIN and self._mode != AgentMode.TRAIN_INFERENCE:
            raise Exception(f"this method is unavailable under the mode {self._mode}")

    def _assert_inference_mode(self):
        if self._mode != AgentMode.INFERENCE and self._mode != AgentMode.TRAIN_INFERENCE:
            raise Exception(f"this method is unavailable under the mode {self._mode}")
