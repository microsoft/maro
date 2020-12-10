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
            session_id=f"ep-{self._scheduler.current_ep}",
            session_type=SessionType.TASK,
            payload={
                PayloadKey.EPISODE: self._scheduler.current_ep,
                PayloadKey.MODEL: model_dict,
                PayloadKey.EXPLORATION_PARAMS: exploration_params
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
        self._actors = self._proxy.peers_name[ACTOR]
        self._pending_actor_set = None
        if choose_action_trigger is None:
            choose_action_trigger = len(self._actors)
        self._registry_table.register_event_handler(
            f"{ACTOR}:{MessageTag.ACTION.value}:{choose_action_trigger}", self._get_action
        )
        if update_trigger is None:
            update_trigger = len(self._actors)
        self._registry_table.register_event_handler(f"{ACTOR}:{MessageTag.UPDATE.value}:{update_trigger}", self._update)
        self._latest_time_steps_by_actor = defaultdict(lambda: -1)

    def learn(self):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        for exploration_params in self._scheduler:
            # load exploration parameters
            if exploration_params:
                self._agent_manager.update_exploration_params(exploration_params)
            self._sample()
            self._serve()

    def test(self):
        """Test policy performance."""
        self._sample(is_training=False)
        self._serve()

    def _sample(self, is_training: bool = True):
        """Send roll-out requests to remote actors.

        Args:
            is_training (bool): If True, return experiences as well as performance metrics provided by the env.

        Returns:
            Performance and per-agent experiences from the remote actor.
        """
        self._proxy.ibroadcast(
            component_type=ACTOR,
            tag=MessageTag.ROLLOUT,
            session_id=f"ep-{self._scheduler.current_ep}",
            session_type=SessionType.TASK,
            payload={PayloadKey.IS_TRAINING: is_training}
        )

    def _serve(self):
        self._pending_actor_set = set(self._actors)
        self._latest_time_steps_by_actor = defaultdict(lambda: -1)
        for msg in self._proxy.receive():
            if msg.tag == MessageTag.ACTION:
                actor_id, ep, time_step = msg.session_id.split(".")
                ep, time_step = int(ep.split("-")[-1]), int(time_step.split("-")[-1])
                if ep == self._scheduler.current_ep and time_step > self._latest_time_steps_by_actor[actor_id]:
                    self._latest_time_steps_by_actor[actor_id] = time_step
                    self._registry_table.push_process(msg)
            elif msg.tag == MessageTag.UPDATE:
                # If enough update messages have been received, call _update() and break out of the loop to start the
                # next episode.
                if self._registry_table.push_process(msg):
                    break

        self._registry_table.clear()
        for actor_id in self._pending_actor_set:
            self._proxy.isend(
                SessionMessage(
                    tag=MessageTag.FORCE_RESET,
                    source=self._proxy.component_name,
                    destination=actor_id,
                    session_id=f"ep-{self._scheduler.current_ep}",
                    session_type=SessionType.NOTIFICATION
                )
            )

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

    def _update(self, messages: list):
        if isinstance(messages, SessionMessage):
            messages = [messages]
            
        for msg in messages:
            self._scheduler.record_performance(msg.payload[PayloadKey.PERFORMANCE])
            self._pending_actor_set.remove(msg.source)

        self._agent_manager.train(
            self._experience_collecting_func({msg.source: msg.payload[PayloadKey.EXPERIENCES] for msg in messages})
        )
