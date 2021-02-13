# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, List, Union

import numpy as np

from maro.communication import Message, Proxy, RegisterTable, SessionType
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.scheduling.scheduler import Scheduler
from maro.utils import InternalLogger

from .message_enums import MessageTag, PayloadKey


class AbsLearner(ABC):
    """Learner class.

    Args:
        group_name (str): Identifier of the group to which the actor belongs. It must be the same group name
            assigned to the actors (and roll-out clients, if any).
        num_actors (int): Expected number of actors in the group idnetified by ``group_name``.
        agent (Union[AbsAgent, MultiAgentWrapper]): Agent or ditionary of agents managed by the agent.
        scheduler (AbsScheduler): A scheduler responsible for iterating over episodes and generating exploration
            parameters if necessary. Defaults to None.
        update_trigger (str): Number or percentage of ``MessageTag.FINISHED`` messages required to trigger
            learner updates, i.e., model training.
        inference (bool): If true, inference (i.e., action decisions) will be performed on the learner side.
            See https://arxiv.org/pdf/1910.06591.pdf for details. Defaults to False.
        inference_trigger (str): Number or percentage of ``MessageTag.CHOOSE_ACTION`` messages required to tigger
            batch inference.
        state_batching_func (Callable): A function to batch state objects from multiple roll-out clients
            for batch inference. Ignored if ``inference`` is false.
        proxy_options (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
    """
    def __init__(
        self,
        group_name: str,
        num_actors: int,
        agent: Union[AbsAgent, MultiAgentWrapper],
        scheduler: Scheduler = None,
        update_trigger: str = None,
        inference: bool = False,
        inference_trigger: str = None,
        state_batching_func: Callable = None,
        proxy_options: dict = None
    ):
        super().__init__()
        self.agent = agent
        self.scheduler = scheduler
        self.inference = inference
        if proxy_options is None:
            proxy_options = {}
        peers = {"actor": num_actors}
        if inference:
            peers["decision_client"] = num_actors
        self._proxy = Proxy(group_name, "learner", peers, **proxy_options)
        self._actors = self._proxy.peers_name["actor"]  # remote actor ID's
        self._registry_table = RegisterTable(self._proxy.peers_name)
        if update_trigger is None:
            update_trigger = len(self._actors)
        self._registry_table.register_event_handler(
            f"actor:{MessageTag.FINISHED.value}:{update_trigger}", self._on_rollout_finish
        )
        if inference:
            self._decision_clients = self._proxy.peers_name["decision_client"]
            if inference_trigger is None:
                inference_trigger = len(self._decision_clients)
            self._registry_table.register_event_handler(
                f"decision_client:{MessageTag.CHOOSE_ACTION.value}:{inference_trigger}", self._on_action_request
            )
            self._state_batching_func = state_batching_func
        else:
            self._decision_clients = None
            self._state_batching_func = None

        self.logger = InternalLogger(self._proxy.component_name)

    @abstractmethod
    def run(self):
        raise NotADirectoryError

    def collect(self, rollout_index: int, training: bool = True, **rollout_kwargs) -> tuple:
        """Collect roll-out performances and details from remote actors.

        Args:
            rollout_index (int): Index of roll-out requests.
            training (bool): If true, the roll-out request is for training purposes.
            rollout_kwargs: Keyword parameters required for roll-out. Must match the keyword parameters specified
                for the roll-out executor.
        """
        payload = {
            PayloadKey.ROLLOUT_INDEX: rollout_index,
            PayloadKey.TRAINING: training,
            PayloadKey.ROLLOUT_KWARGS: rollout_kwargs
        }
        # If no actor client is found, it is necessary to broadcast agent models to the remote actors
        # so that thay can perform inference on their own. If there exists exploration parameters, they
        # must also be sent to the remote actors.
        self._proxy.iscatter(MessageTag.ROLLOUT, SessionType.TASK, [(actor, payload) for actor in self._actors])
        self.logger.info(f"Sent roll-out requests to {self._actors} for ep-{rollout_index}")

        # Receive roll-out results from remote actors
        for msg in self._proxy.receive():
            if msg.payload[PayloadKey.ROLLOUT_INDEX] != rollout_index:
                self.logger.info(
                    f"Ignore a message of type {msg.tag} with ep {msg.payload[PayloadKey.ROLLOUT_INDEX]} "
                    f"(current ep: {rollout_index})"
                )
                continue
            if msg.tag == MessageTag.FINISHED:
                # If enough update messages have been received, call update() and break out of the loop to start
                # the next episode.
                result = self._registry_table.push(msg)
                if result:
                    env_metrics, details = result[0]
                    break
            elif msg.tag == MessageTag.CHOOSE_ACTION:
                self._registry_table.push(msg)

        return env_metrics, details

    def _on_rollout_finish(self, messages: List[Message]):
        metrics = {msg.source: msg.payload[PayloadKey.METRICS] for msg in messages}
        details = {msg.source: msg.payload[PayloadKey.DETAILS] for msg in messages}
        return metrics, details

    def _on_action_request(self, messages: List[Message]):
        # group messages from different actors by the AGENT_ID field
        if isinstance(self.agent, MultiAgentWrapper):
            queries_by_agent_id = defaultdict(list)
            for msg in messages:
                queries_by_agent_id[msg.payload[PayloadKey.AGENT_ID]].append(msg)

            # batch inference for each agent_id
            for agent_id, queries in queries_by_agent_id.items():
                state_batch = self._state_batching_func([query.payload[PayloadKey.STATE] for query in queries])
                action_info = self.agent[agent_id].choose_action(state_batch)
                self._serve(queries, action_info)
        else:
            state_batch = self._state_batching_func([msg.payload[PayloadKey.STATE] for msg in messages])
            action_info = self.agent.choose_action(state_batch)
            self._serve(messages, action_info)

    def _serve(self, queries, action_info):
        if isinstance(action_info, tuple):
            action_info = list(zip(*action_info))
        for query, action in zip(queries, action_info):
            if np.random.random() < 0.99:
                self._proxy.reply(
                    query,
                    tag=MessageTag.ACTION,
                    payload={
                        PayloadKey.ACTION: action,
                        PayloadKey.ROLLOUT_INDEX: query.payload[PayloadKey.ROLLOUT_INDEX],
                        PayloadKey.TIME_STEP: query.payload[PayloadKey.TIME_STEP]
                    }
                )

    def exit(self):
        """Tell the remote actor to exit."""
        self._proxy.ibroadcast(
            component_type="actor", tag=MessageTag.EXIT, session_type=SessionType.NOTIFICATION
        )
        self.logger.info("Exiting...")
        sys.exit(0)
