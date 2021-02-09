# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, List, Union

from maro.communication import Message, Proxy, RegisterTable, SessionType
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.scheduling.scheduler import Scheduler
from maro.utils import InternalLogger

from .message_enums import MessageTag, PayloadKey


class AbsLearner(ABC):
    """Learner class.

    Args:
        agent (Union[AbsAgent, MultiAgentWrapper]): Agent or ditionary of agents managed by the agent.
        proxy: A ``Proxy`` instance responsible for communication.
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
    """
    def __init__(
        self,
        agent: Union[AbsAgent, MultiAgentWrapper],
        proxy: Proxy,
        scheduler: Scheduler = None,
        update_trigger: str = None,
        inference: bool = False,
        inference_trigger: str = None,
        state_batching_func: Callable = None
    ):
        super().__init__()
        self.agent = agent
        self.proxy = proxy
        self.scheduler = scheduler
        self._actors = self.proxy.peers_name["actor"]  # remote actor ID's
        self._registry_table = RegisterTable(self.proxy.peers_name)
        if update_trigger is None:
            update_trigger = len(self._actors)
        self._registry_table.register_event_handler(
            f"actor:{MessageTag.FINISHED.value}:{update_trigger}", self._on_rollout_finish)
        if inference:
            self._rollout_clients = self.proxy.peers_name["rollout_client"]
            if inference_trigger is None:
                inference_trigger = len(self._rollout_clients)
            self._registry_table.register_event_handler(
                f"rollout_client:{MessageTag.CHOOSE_ACTION.value}:{inference_trigger}", self._on_action_request
            )
            self._state_batching_func = state_batching_func
        else:
            self._rollout_clients = None
            self._state_batching_func = None
        self.logger = InternalLogger(self.proxy.component_name)

    @abstractmethod
    def run(self):
        raise NotADirectoryError

    def collect(
        self,
        rollout_index: int,
        training: bool = True,
        agents_to_update: Union[str, List[str]] = None,
        exploration_params=None,
        **rollout_kwargs
    ) -> tuple:
        """Collect roll-out performances and details from remote actors.

        Args:
            rollout_index (int): Index of roll-out requests.
            training (bool): If true, the roll-out request is for training purposes.
            agents_to_update (Union[str, List[str]]): ID's of the agents whose models are to be broadcast to
                the actors. If action decisions are made on the learner's side, this is ignored. Otherwise
                defaults to None, in which case all agents' models will be sent.
            exploration_params: Exploration parameters for the actors to use during roll-out. If action decisions
                are made on the learner's side, this is ignored.
            rollout_kwargs: Keyword parameters required for roll-out. Must match the keyword parameters specified
                for the actor class.
        """
        payload = {
            PayloadKey.ROLLOUT_INDEX: rollout_index,
            PayloadKey.TRAINING: training,
            PayloadKey.ROLLOUT_KWARGS: rollout_kwargs
        }
        # If no actor client is found, it is necessary to broadcast agent models to the remote actors
        # so that thay can perform inference on their own. If there exists exploration parameters, they
        # must also be sent to the remote actors.
        if not self._rollout_clients:
            if exploration_params:
                payload[PayloadKey.EXPLORATION_PARAMS] = exploration_params
            payload[PayloadKey.MODEL] = self.agent.dump_model(agent_ids=agents_to_update)
        self.proxy.iscatter(MessageTag.ROLLOUT, SessionType.TASK, [(actor, payload) for actor in self._actors])
        self.logger.info(f"Sent roll-out requests to {self._actors} for ep-{rollout_index}")

        # Receive roll-out results from remote actors
        for msg in self.proxy.receive():
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
            self.proxy.reply(
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
        self.proxy.ibroadcast(
            component_type="actor", tag=MessageTag.EXIT, session_type=SessionType.NOTIFICATION
        )
        self.logger.info("Exiting...")
        sys.exit(0)
