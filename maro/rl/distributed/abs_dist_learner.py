# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Union

import numpy as np

from maro.communication import Message, Proxy, RegisterTable, SessionType
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.scheduling.scheduler import Scheduler
from maro.utils import InternalLogger

from .common import MessageTag, PayloadKey


class AbsDistLearner(ABC):
    """Distributed learner class.

    Args:
        agent (Union[AbsAgent, MultiAgentWrapper]): Agent or ditionary of agents managed by the agent.
        scheduler (AbsScheduler): A scheduler responsible for iterating over episodes and generating exploration
            parameters if necessary.
        proxy: A ``Proxy`` instance responsible for communication.
        update_trigger (str): Number or percentage of ``MessageTag.FINISHED`` messages required to trigger
            learner updates, i.e., model training.
        inference (bool): If true, inference (i.e., action decisions) will be performed on the learner side.
            See https://arxiv.org/pdf/1910.06591.pdf for details. Defaults to False.
        inference_trigger (str): Number or percentage of ``MessageTag.CHOOSE_ACTION`` messages required to tigger
            batch inference.
    """
    def __init__(
        self,
        agent: Union[AbsAgent, MultiAgentWrapper],
        scheduler: Scheduler,
        proxy: Proxy,
        update_trigger: str = None,
        inference: bool = False,
        inference_trigger: str = None
    ):
        super().__init__()
        self.agent = agent
        self.scheduler = scheduler
        self.proxy = proxy
        self._actors = self.proxy.peers_name["actor"]  # remote actor ID's
        self._registry_table = RegisterTable(self.proxy.peers_name)
        if update_trigger is None:
            update_trigger = len(self._actors)
        self._registry_table.register_event_handler(
            f"actor:{MessageTag.FINISHED.value}:{update_trigger}", self._on_rollout_finish)
        if inference:
            self._actor_clients = self.proxy.peers_name["actor_client"]
            if inference_trigger is None:
                inference_trigger = len(self._actor_clients)
            self._registry_table.register_event_handler(
                f"actor_client:{MessageTag.CHOOSE_ACTION.value}:{inference_trigger}", self._on_action_request
            )
        else:
            self._actor_clients = None
        self._logger = InternalLogger(self.proxy.component_name)

    @abstractmethod
    def learn(self):
        return NotImplemented

    @abstractmethod
    def update(self, rollout_data):
        return NotImplemented

    def collect(
        self,
        is_training: bool = True,
        agents_to_update: Union[str, List[str]] = None,
        exploration_params=None,
        **rollout_kwargs
    ) -> tuple:
        """Collect roll-out performances and details from remote actors.

        Args:
            is_training (bool): If true, the roll-out request is for training purposes.
            agents_to_update (Union[str, List[str]]): ID's of the agents whose models are to be broadcast to
                the actors. If action decisions are made on the learner's side, this is ignored. Otherwise
                defaults to None, in which case all agents' models will be sent.
            exploration_params: Exploration parameters for the actors to use during roll-out. If action decisions
                are made on the learner's side, this is ignored.
        """
        ep = self.scheduler.iter
        payload = {
            PayloadKey.EPISODE: ep,
            PayloadKey.IS_TRAINING: is_training,
            PayloadKey.ROLLOUT_KWARGS: rollout_kwargs
        }
        # If no actor client is found, it is necessary to broadcast agent models to the remote actors
        # so that thay can perform inference on their own. If there exists exploration parameters, they
        # must also be sent to the remote actors.   
        if not self._actor_clients:
            if exploration_params:
                payload[PayloadKey.EXPLORATION_PARAMS] = exploration_params
            payload[PayloadKey.MODEL] = self.agent.dump_model(agent_ids=agents_to_update)
        self.proxy.iscatter(MessageTag.ROLLOUT, SessionType.TASK, [(actor, payload) for actor in self._actors])
        self._logger.info(f"Sent roll-out requests to {self._actors} for ep-{ep}")
        
        # Receive roll-out results from remote actors
        unfinished = set(self._actors) if not self._actor_clients else set(self._actor_clients)
        for msg in self.proxy.receive():
            if msg.payload[PayloadKey.EPISODE] != ep:
                self._logger.info(
                    f"Ignore a message of {msg.tag} with ep {msg.payload[PayloadKey.EPISODE]} (current ep: {ep})"
                )
                continue
            if msg.tag == MessageTag.FINISHED:
                # If enough update messages have been received, call update() and break out of the loop to start
                # the next episode.
                src = msg.payload[PayloadKey.ACTOR_CLIENT_ID] if self._actor_clients else msg.source
                unfinished.discard(src)
                result = self._registry_table.push(msg)
                if result:
                    performance, details = result[0]
                    break
            elif msg.tag == MessageTag.CHOOSE_ACTION:
                self._registry_table.push(msg)

        # Send a TERMINATE_EPISODE cmd to unfinished actors to catch them up.
        if unfinished:
            self.terminate_rollout(list(unfinished))

        return performance, details

    def terminate_rollout(self, actors: List[str]):
        """Send messages to select actors to terminate their roll-out routines."""
        self.proxy.iscatter(
            MessageTag.TERMINATE_EPISODE, SessionType.NOTIFICATION,
            [(name, {PayloadKey.EPISODE: self.scheduler.iter}) for name in actors]
        )
        self._logger.info(f"Sent terminating signals to unfinished actors: {actors}")

    def _on_rollout_finish(self, finish_messages: List[Message]):
        performances = {msg.source: msg.payload[PayloadKey.PERFORMANCE] for msg in finish_messages}
        details = {msg.source: msg.payload[PayloadKey.DETAILS] for msg in finish_messages}
        return performances, details

    def _on_action_request(self, messages: List[Message]):
        # group messages from different actors by the AGENT_ID field
        messages_by_agent_id = defaultdict(list)
        for msg in messages:
            messages_by_agent_id[msg.payload[PayloadKey.AGENT_ID]].append(msg)

        # batch inference for each agent_id
        for agent_id, message_batch in messages_by_agent_id.items():
            state_batch = np.vstack([msg.payload[PayloadKey.STATE] for msg in message_batch])
            action_batch = self.agent[agent_id].choose_action(state_batch)
            for msg, action in zip(message_batch, action_batch):
                self.proxy.reply(
                    msg,
                    tag=MessageTag.ACTION,
                    payload={
                        PayloadKey.ACTION: action,
                        PayloadKey.EPISODE: msg.payload[PayloadKey.EPISODE],
                        PayloadKey.TIME_STEP: msg.payload[PayloadKey.TIME_STEP]
                    }
                )
    
    def exit(self):
        """Tell the remote actor to exit."""
        self.proxy.ibroadcast(
            component_type="actor", tag=MessageTag.EXIT, session_type=SessionType.NOTIFICATION
        )
        self._logger.info("Exiting...")
        sys.exit(0)
