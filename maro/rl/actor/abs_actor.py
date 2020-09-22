# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Union

from maro.rl.agent.abs_agent_manager import AbsAgentManager
from maro.simulator import Env


class AbsActor(ABC):
    def __init__(self, env: Env, inference_agents: Union[dict, AbsAgentManager]):
        """
        Actor contains env and agents, and it is responsible for collecting experience from the interaction
        with the environment.

        Args:
            env (Env): an Env instance.
            inference_agents (dict or AbsAgentManager): a dict of agents or an AgentManager instance that
                                                        manages all agents.
        """
        self._env = env
        self._inference_agents = inference_agents

    @abstractmethod
    def roll_out(self, model_dict: dict = None, epsilon_dict: dict = None, done: bool = None,
                 return_details: bool = True):
        """
        Performs a single episode of roll-out to collect experiences and performance data from the environment.

        Args:
            model_dict (dict): if not None, the agents will load the models from model_dict and use these models
                           to perform roll-out.
            epsilon_dict (dict): exploration rate by agent.
            done (bool): if True, the current call is the last call, i.e., no more roll-outs will be performed.
                         This flag is used to signal remote actor workers to exit.
            return_details (bool): if True, return episode details (e.g., experiences) as well as performance
                                   metrics provided by the env.
        """
        return NotImplementedError

    @property
    def inference_agents(self):
        return self._inference_agents
