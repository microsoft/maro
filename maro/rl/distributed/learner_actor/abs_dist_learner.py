# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from abc import ABC, abstractmethod
from typing import Callable

from maro.communication import Proxy, RegisterTable, SessionType
from maro.rl.agent.abs_agent_manager import AbsAgentManager
from maro.rl.scheduling.scheduler import Scheduler
from maro.utils import DummyLogger, Logger

from .common import Component, MessageTag


class AbsDistLearner(ABC):
    """Abstract distributed learner class.

    Args:
        agent_manager (AbsAgentManager): An AgentManager instance that manages all agents.
        scheduler (AbsScheduler): A scheduler responsible for iterating over episodes and generating exploration
            parameters if necessary.
        experience_collection_func (Callable): Function to collect experiences from multiple remote actors.
        logger (Logger): Used to log important messages.
        proxy_params: Parameters required for instantiating an internal proxy for communication.
    """
    def __init__(
        self,
        agent_manager: AbsAgentManager,
        scheduler: Scheduler,
        experience_collecting_func: Callable,
        logger: Logger = DummyLogger(),
        **proxy_params
    ):
        super().__init__()
        self._agent_manager = agent_manager
        self._scheduler = scheduler
        self._experience_collecting_func = experience_collecting_func
        self._proxy = Proxy(component_type=Component.LEARNER.value, **proxy_params)
        self._registry_table = RegisterTable(self._proxy.peers_name)
        self._logger = logger

    @abstractmethod
    def learn(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        return NotImplementedError

    def exit(self):
        """Tell the remote actor to exit."""
        self._proxy.ibroadcast(
            component_type=Component.ACTOR.value, tag=MessageTag.EXIT, session_type=SessionType.NOTIFICATION
        )
        sys.exit(0)

    def load_models(self, dir_path: str):
        self._agent_manager.load_models_from_files(dir_path)

    def dump_models(self, dir_path: str):
        self._agent_manager.dump_models_to_files(dir_path)
