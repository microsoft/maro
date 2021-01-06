# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import List, Union

import numpy as np

from maro.communication import SessionMessage
from maro.utils import DummyLogger, Logger

from .abs_dist_learner import AbsDistLearner
from .common import Component, MessageTag, PayloadKey

ACTOR = Component.ACTOR.value


class SimpleDistLearner(AbsDistLearner):
    """Simple distributed learner that broadcasts models to remote actors for roll-out purposes."""
    def learn(self):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        for exploration_params in self._scheduler:
            self._current_stage = str(self._scheduler.current_ep)
            # set exploration parameters
            if exploration_params:
                self._agent_manager.set_exploration_params(exploration_params)
            self._request_rollout()
            self._wait_for_actor_results()

        self._proxy.clear_cache()

    def test(self):
        """Test policy performance on remote actors."""
        self._current_stage = "test"
        self._request_rollout()
        self._wait_for_actor_results()
        self._proxy.clear_cache()

    def _wait_for_actor_results(self):
        """Wait for roll-out results from remote actors."""
        for msg in self._proxy.receive():
            if (msg.tag == MessageTag.FINISHED and 
                    msg.session_id == self._current_stage and
                    self._registry_table.push_process(msg)
            ):
                # If enough update messages have been received, call _update() and break out of the loop to start the
                # next episode.
                break


class InferenceLearner(AbsDistLearner):
    """Distributed learner based on SEED RL architecture.

    See https://arxiv.org/pdf/1910.06591.pdf for experiences.
    """
    def __init__(
        self,
        agent_manager,
        scheduler,
        experience_collecting_func,
        choose_action_trigger: str = None,
        update_trigger: str = None,
        **proxy_params
    ):
        super().__init__(
            agent_manager, scheduler, experience_collecting_func, update_trigger=update_trigger, **proxy_params
        )
        if choose_action_trigger is None:
            choose_action_trigger = len(self._actors)
        self._registry_table.register_event_handler(
            f"{ACTOR}:{MessageTag.ACTION.value}:{choose_action_trigger}", self._get_action
        )
        self._latest_time_steps_by_actor = defaultdict(lambda: -1)

    def learn(self):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        self._is_training = True
        for exploration_params in self._scheduler:
            self._current_stage = str(self._scheduler.current_ep)
            # set exploration parameters
            if exploration_params:
                self._agent_manager.set_exploration_params(exploration_params)
            self._request_rollout()
            self._serve_and_update()

        self._proxy.clear_cache()

    def test(self):
        """Test policy performance on remote actors."""
        self._current_stage = "test"
        self._request_rollout()
        self._serve_and_update()
        self._proxy.clear_cache()

    def _serve_and_update(self):
        """Serve actions to actors and wait for their roll-out results."""
        self._latest_time_steps_by_actor = defaultdict(lambda: -1)

        for msg in self._proxy.receive():
            if msg.tag == MessageTag.ACTION:
                actor_id = msg.source
                stage, time_step = msg.session_id.split(".")
                time_step = int(time_step)
                if stage == self._current_stage and time_step > self._latest_time_steps_by_actor[actor_id]:
                    self._latest_time_steps_by_actor[actor_id] = time_step
                    self._registry_table.push_process(msg)
            elif (msg.tag == MessageTag.FINISHED and
                    msg.session_id == self._current_stage and
                    self._registry_table.push_process(msg)
            ):
                # If enough update messages have been received, call _update() and break out of the loop to start the
                # next episode.
                break

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
                self._proxy.reply(received_message=msg, tag=MessageTag.ACTION, payload={PayloadKey.ACTION: action})
