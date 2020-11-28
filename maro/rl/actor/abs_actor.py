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
        agents (AbsAgentManager or dict): A dict of agents or an AgentManager instance that manages
            all agents.
    """
    def __init__(self, env: Env, agents: Union[AbsAgentManager, dict]):
        self._env = env
        self._agents = agents

    @abstractmethod
    def roll_out(self, *args, **kwargs):
        """This method performs a single episode of roll-out."""
        return NotImplementedError

    @property
    def agents(self):
        """Agents performing inference during roll-out."""
        return self._agents
