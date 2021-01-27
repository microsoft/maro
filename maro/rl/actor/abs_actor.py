# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

from maro.rl.agent_manager.abs_agent_manager import AbsAgentManager
from maro.simulator import Env


class AbsActor(ABC):
    """Abstract actor class.

    An actor is a wrapper of an env and an agent manager or a dict of agents and is responsible for collecting
    experience from interacting with the environment.

    Args:
        env (Env): An Env instance.
        agent_manager (AbsAgentManager): An AgentManager instance that contains necessary shapers and manages
            all agents.
    """
    def __init__(self, env: Env, agent_manager: AbsAgentManager):
        self._env = env
        self.agent_manager = agent_manager

    @abstractmethod
    def roll_out(
        self, model_dict: dict = None, exploration_params=None, done: bool = None, return_details: bool = True
    ):
        """This method performs a single episode of roll-out.

        Args:
            model_dict (dict): If not None, the agents will load the models from model_dict and use these models
                to perform roll-out.
            exploration_params: Exploration parameters.
            done (bool): If True, the current call is the last call, i.e., no more roll-outs will be performed.
                This flag is used to signal remote actor workers to exit.
            return_details (bool): If True, return episode details (e.g., experiences) as well as performance
                metrics provided by the env.

        Returns:
            Relevant results from the roll-out (e.g., performance, experiences), depending on the implementation.
        """
        return NotImplementedError
