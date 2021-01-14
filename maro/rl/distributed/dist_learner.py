# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import List, Union

import numpy as np

from maro.communication import SessionMessage, SessionType
from maro.utils import Logger

from .abs_dist_learner import AbsDistLearner
from .common import MessageTag, PayloadKey, TerminateEpisode


class SimpleDistLearner(AbsDistLearner):
    """Simple distributed learner that broadcasts models to remote actors for roll-out purposes."""
    def learn(self):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        for exploration_params in self._scheduler:
            # set exploration parameters
            if exploration_params:
                self._agent_manager.set_exploration_params(exploration_params)
            self._request_rollout()
            self._wait_for_actor_results()

    def test(self):
        """Test policy performance on remote actors."""
        self._request_rollout(is_training=False)
        self._wait_for_actor_results()

    def _wait_for_actor_results(self):
        """Wait for roll-out results from remote actors."""
        ep = self._scheduler.current_ep if is_training else "test"
        unfinished = set(self._actors)
        for msg in self._proxy.receive():
            if msg.tag == MessageTag.FINISHED:
                if msg.payload[PayloadKey.EPISODE] != ep:
                    self._logger.info(
                        f"Ignore a message of {msg.tag} with ep {msg.payload[PayloadKey.EPISODE]} (current ep: {ep})"
                    )
                    continue
                unfinished.discard(msg.source)
                if self._registry_table.push(msg) is not None:
                # If enough update messages have been received, call _update() and break out of the loop to start the
                # next episode.
                    break

        # Send a TERMINATE_EPISODE cmd to unfinished actors to catch them up.
        if unfinished:
            self._proxy.iscatter(
                MessageTag.TERMINATE_EPISODE, SessionType.NOTIFICATION,
                [(actor, {PayloadKey.EPISODE: ep}) for actor in unfinished]
            )
            self._logger.info(f"Sent terminating signals to unfinished actors: {unfinished}")


class InferenceLearner(AbsDistLearner):
    """Distributed learner based on SEED RL architecture.

    See https://arxiv.org/pdf/1910.06591.pdf for experiences.
    """
    def __init__(
        self,
        agent_manager,
        scheduler,
        proxy,
        experience_collecting_func,
        choose_action_trigger: str = None,
        update_trigger: str = None
    ):
        super().__init__(
            agent_manager, scheduler, proxy, experience_collecting_func, update_trigger=update_trigger
        )
        self._peer_agent_managers = self._proxy.peers_name["agent_manager"]
        if choose_action_trigger is None:
            choose_action_trigger = len(self._actors)
        self._registry_table.register_event_handler(
            f"agent_manager:{MessageTag.CHOOSE_ACTION.value}:{choose_action_trigger}", self._get_action
        )

    def learn(self):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        for exploration_params in self._scheduler:
            # set exploration parameters
            if exploration_params:
                self._agent_manager.set_exploration_params(exploration_params)
            self._request_rollout(with_model_copies=False)
            self._serve_and_update()

    def test(self):
        """Test policy performance on remote actors."""
        self._request_rollout(is_training=False, with_model_copies=False)
        self._serve_and_update(is_training=False)

    def _serve_and_update(self, is_training: bool = True):
        """Serve actions to actors and wait for their roll-out results."""
        ep = self._scheduler.current_ep if is_training else "test"
        unfinished = set(self._peer_agent_managers)
        for msg in self._proxy.receive():
            if msg.payload[PayloadKey.EPISODE] != ep:
                self._logger.info(
                    f"Ignore a message of {msg.tag} with ep {msg.payload[PayloadKey.EPISODE]} (current ep: {ep})")
                continue
            if msg.tag == MessageTag.FINISHED:
                # If enough update messages have been received, call _update() and break out of the loop to start
                # the next episode.
                unfinished.discard(msg.payload[PayloadKey.AGENT_MANAGER_ID])
                if self._registry_table.push(msg) is not None:
                    break
            elif msg.tag == MessageTag.CHOOSE_ACTION:
                self._registry_table.push(msg)

        # Send a TERMINATE_EPISODE cmd to unfinished actors to catch them up.
        if unfinished:
            self._proxy.iscatter(
                MessageTag.TERMINATE_EPISODE, SessionType.NOTIFICATION,
                [(actor, {PayloadKey.EPISODE: ep}) for actor in unfinished]
            )
            self._logger.info(f"Sent terminating signals to unfinished actors: {unfinished}")

    def _get_action(self, messages: Union[List[SessionMessage], SessionMessage]):
        if isinstance(messages, SessionMessage):
            messages = [messages]

        # group messages from different actors by the AGENT_ID field
        messages_by_agent_id = defaultdict(list)
        for msg in messages:
            messages_by_agent_id[msg.payload[PayloadKey.AGENT_ID]].append(msg)

        # batch inference for each agent_id
        for agent_id, message_batch in messages_by_agent_id.items():
            state_batch = np.vstack([msg.payload[PayloadKey.STATE] for msg in message_batch])
            action_batch = self._agent_manager[agent_id].choose_action(state_batch)
            for msg, action in zip(message_batch, action_batch):
                self._proxy.reply(
                    msg,
                    tag=MessageTag.ACTION,
                    payload={
                        PayloadKey.ACTION: action,
                        PayloadKey.EPISODE: msg.payload[PayloadKey.EPISODE],
                        PayloadKey.TIME_STEP: msg.payload[PayloadKey.TIME_STEP]
                    }
                )
