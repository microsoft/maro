# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from abc import ABC, abstractmethod
from typing import Callable

from maro.communication import Proxy, RegisterTable, SessionMessage, SessionType
from maro.rl.agent_manager import AbsAgentManager
from maro.rl.scheduling.scheduler import Scheduler
from maro.utils import InternalLogger

from .common import MessageTag, PayloadKey


class AbsDistLearner(ABC):
    """Abstract distributed learner class.

    Args:
        agent_manager (AbsAgentManager): An AgentManager instance that manages all agents.
        scheduler (AbsScheduler): A scheduler responsible for iterating over episodes and generating exploration
            parameters if necessary.
        proxy: A ``Proxy`` instance responsible for communication.
        experience_collection_func (Callable): Function to collect experiences from multiple remote actors.
        update_trigger (str): Number or percentage of ``MessageTag.FINISHED`` messages required to trigger
            the ``_update`` method, i.e., model training.
    """
    def __init__(
        self,
        agent_manager: AbsAgentManager,
        scheduler: Scheduler,
        proxy: Proxy,
        experience_collecting_func: Callable,
        update_trigger: str = None,
        **proxy_params
    ):
        super().__init__()
        self._agent_manager = agent_manager
        self._scheduler = scheduler
        self._experience_collecting_func = experience_collecting_func
        self._proxy = proxy
        self._registry_table = RegisterTable(self._proxy.peers_name)
        self._actors = self._proxy.peers_name["actor"]
        if update_trigger is None:
            update_trigger = len(self._actors)
        self._registry_table.register_event_handler(
            f"actor:{MessageTag.FINISHED.value}:{update_trigger}", self._update
        )
        self._logger = InternalLogger(self._proxy.component_name)

    @abstractmethod
    def learn(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        return NotImplementedError

    def exit(self):
        """Tell the remote actor to exit."""
        self._proxy.ibroadcast(
            component_type="actor", tag=MessageTag.EXIT, session_type=SessionType.NOTIFICATION
        )
        self._logger.info("Exiting...")
        sys.exit(0)

    def load_models(self, dir_path: str):
        self._agent_manager.load_models_from_files(dir_path)

    def dump_models(self, dir_path: str):
        self._agent_manager.dump_models_to_files(dir_path)

    def _request_rollout(self, is_training: bool = True, with_model_copies: bool = True):
        """Send roll-out requests to remote actors."""
        ep = self._scheduler.current_ep if is_training else "test"
        payload = {PayloadKey.EPISODE: ep, PayloadKey.IS_TRAINING: is_training}
        if with_model_copies:
            payload[PayloadKey.MODEL] = self._agent_manager.dump_models()
        self._proxy.iscatter(MessageTag.ROLLOUT, SessionType.TASK, [(actor, payload) for actor in self._actors])
        self._logger.info(f"Sent roll-out requests to {self._actors} for ep-{ep}")

    def _update(self, messages: list):
        if isinstance(messages, SessionMessage):
            messages = [messages]

        is_training = messages[0].payload[PayloadKey.EXPERIENCES] is not None
        for msg in messages:
            performance = msg.payload[PayloadKey.PERFORMANCE]
            self._scheduler.record_performance(performance)
            current_ep = self._scheduler.current_ep if is_training else "test"
            self._logger.info(
                f"{msg.source}.ep-{current_ep} - performance: {performance}, "
                f"exploration_params: {self._scheduler.exploration_params if is_training else None}"
            )

        # If the learner is in training mode, perform model updates.
        if is_training:
            experiences = self._experience_collecting_func(
                {msg.source: msg.payload[PayloadKey.EXPERIENCES] for msg in messages}
            )
            self._agent_manager.train(experiences)
            self._logger.info("Training finished")

        self._registry_table.clear()
