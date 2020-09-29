# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Union

from maro.rl.agent.abs_agent_manager import AbsAgentManager
from maro.simulator import Env


class AbsActor(ABC):
    """Abstract actor class.

    An actor is a wrapper of an env and an agent manager or a dict of agents and is responsible for collecting
    experience from interacting with the environment.

    Args:
        env (Env): An Env instance.
        inference_agents (AbsAgentManager or dict): A dict of agents or an AgentManager instance that manages
            all agents.
    """
    def __init__(self, env: Env, inference_agents: Union[AbsAgentManager, dict]):
        self._env = env
        self._inference_agents = inference_agents

    @abstractmethod
    def roll_out(self, model_dict: dict = None, epsilon_dict: dict = None, done: bool = None,
                 return_details: bool = True):
        """This method performs a single episode of roll-out.

        Args:
            model_dict (dict): If not None, the agents will load the models from model_dict and use these models
                to perform roll-out.
            epsilon_dict (dict): Exploration rate by agent.
            done (bool): If True, the current call is the last call, i.e., no more roll-outs will be performed.
                This flag is used to signal remote actor workers to exit.
            return_details (bool): If True, return episode details (e.g., experiences) as well as performance
                metrics provided by the env.

        Returns:
            Relevant results from the roll-out (e.g., performance, experiences), depending on the implementation.
        """
        return NotImplementedError

    @property
    def inference_agents(self):
        """Agents performing inference during roll-out."""
        return self._inference_agents
