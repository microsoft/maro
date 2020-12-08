# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import List, Union

import numpy as np

from maro.communication import SessionMessage, SessionType

from .abs_dist_learner import AbsDistLearner
from .common import Component, MessageTag, PayloadKey

ACTOR = Component.ACTOR.value


class SimpleDistLearner(AbsDistLearner):
    """Simple distributed learner that broadcasts models to remote actors for roll-out purposes."""
    def learn(self):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        for exploration_params in self._scheduler:
            performance, exp_by_agent = self._sample(
                self._agent_manager.dump_models(), exploration_params=exploration_params
            )
            self._scheduler.record_performance(performance)
            self._agent_manager.train(exp_by_agent)

    def test(self):
        """Test policy performance."""
        performance, _ = self._sample(self._agent_manager.dump_models(), return_experiences=False)
        self._scheduler.record_performance(performance)

    def _sample(self, model_dict: dict, exploration_params=None, return_experiences: bool = True):
        """Send roll-out requests to remote actors.

        Args:
            model_dict (dict): Models the remote actors .
            exploration_params: Exploration parameters.
            return_experiences (bool): If True, return experiences as well as performance metrics provided by the env.

        Returns:
            Performance and per-agent experiences from the remote actor.
        """
        # TODO: double check when ack enable
        replies = self._proxy.broadcast(
            component_type=ACTOR,
            tag=MessageTag.ROLLOUT,
            session_id=".".join([f"ep_{self._scheduler.current_ep}", "roll_out"]),
            session_type=SessionType.TASK,
            payload={
                PayloadKey.EPISODE: self._scheduler.current_ep,
                PayloadKey.MODEL: model_dict,
                PayloadKey.EXPLORATION_PARAMS: exploration_params,
                PayloadKey.RETURN_EXPERIENCES: return_experiences
            }
        )

        performance = [(msg.source, msg.payload[PayloadKey.PERFORMANCE]) for msg in replies]
        experiences_by_source = {msg.source: msg.payload[PayloadKey.EXPERIENCES] for msg in replies}
        experiences = self._experience_collecting_func(experiences_by_source) if return_experiences else None

        return performance, experiences


class SEEDLearner(AbsDistLearner):
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
        super().__init__(agent_manager, scheduler, experience_collecting_func, **proxy_params)
        self._num_actors = len(self._proxy.peers_name[ACTOR])
        if choose_action_trigger is None:
            choose_action_trigger = self._num_actors
        self._registry_table.register_event_handler(
            f"{ACTOR}:{MessageTag.CHOOSE_ACTION.value}:{choose_action_trigger}", self._get_action
        )
        if update_trigger is None:
            update_trigger = self._num_actors
        self._registry_table.register_event_handler(
            f"{ACTOR}:{MessageTag.UPDATE.value}:{update_trigger}", self._collect
        )
        self._experiences = {}
        self._rollout_complete_counter = 0

    def learn(self):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        for exploration_params in self._scheduler:
            self._rollout_complete_counter = 0
            self._experiences.clear()
            # load exploration parameters:
            if exploration_params is not None:
                self._agent_manager.update_exploration_params(exploration_params)
            self._sample()
            self._serve()
            self._agent_manager.train(self._experience_collecting_func(self._experiences))

    def test(self):
        """Test policy performance."""
        self._sample()
        self._serve()

    def _sample(self, return_experiences: bool = True):
        """Send roll-out requests to remote actors.

        Args:
            return_experiences (bool): If True, return experiences as well as performance metrics provided by the env.

        Returns:
            Performance and per-agent experiences from the remote actor.
        """
        self._proxy.ibroadcast(
            component_type=ACTOR,
            tag=MessageTag.ROLLOUT,
            session_id=".".join([f"ep_{self._scheduler.current_ep}", "roll_out"]),
            session_type=SessionType.TASK,
            payload={PayloadKey.EPISODE: self._scheduler.current_ep, PayloadKey.RETURN_EXPERIENCES: return_experiences}
        )

    def _serve(self):
        for msg in self._proxy.receive():
            self._registry_table.push(msg)
            for handler_fn, cached_messages in self._registry_table.get():
                handler_fn(cached_messages)
            if self._rollout_complete_counter == self._num_actors:
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

    def _collect(self, messages: list):
        for msg in messages:
            self._scheduler.record_performance(msg.payload[PayloadKey.PERFORMANCE])

        self._experiences = {msg.source: msg.payload[PayloadKey.EXPERIENCES] for msg in messages}
        self._rollout_complete_counter = len(messages)
