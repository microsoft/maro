# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from abc import ABC, abstractmethod
from typing import Union

from maro.simulator import Env
from maro.rl.agent.agent_manager import AgentManager


class RolloutMode(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"
    EXIT = "exit"  # signals a remote actor process to quite


class AbstractActor(ABC):
    def __int__(self, env: Union[dict, Env], inference_agents: Union[dict, AgentManager]):
        """
        Actor contains env and agents, with the main task being environment sampling for experience collection
        Args:
            env (Env): an Env instance
            inference_agents (dict or AgentManager): a dict of agents or an AgentManager instance that manages all agents
        """
        self._env = env
        self._inference_agents = inference_agents

    @abstractmethod
    def roll_out(self, mode: RolloutMode, models: dict = None, epsilon_dict: dict = None):
        """
        The main interface provided by the Actor class, in which the agents perform a single episode of roll-out
        to collect experiences and performance data from the environment
        Args:
            mode (RolloutMode): determines the type of rollout task to be performed
            models (dict): if not None, the agents will load the models from model_dict and use these models
                           to perform roll-out.
            epsilon_dict (dict): exploration rate
        """
        return NotImplementedError

    @property
    def inference_agents(self):
        return self._inference_agents
