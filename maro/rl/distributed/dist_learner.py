# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.communication import SessionType
from maro.rl.distributed.common import MessageTag, PayloadKey

from .abs_dist_learner import AbsDistLearner


class SimpleDistLearner(AbsDistLearner):
    """Distributed"""
    def learn(self):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        for exploration_params in self._scheduler:
            performance, exp_by_agent = self._sample(
                self._agent_manager.dump_models(),
                exploration_params=exploration_params
            )
            self._scheduler.record_performance(performance)
            self._agent_manager.train(exp_by_agent)

    def test(self):
        """Test policy performance."""
        performance, _ = self._sample(self._agent_manager.dump_models(), return_details=False)
        self._scheduler.record_performance(performance)

    def _sample(self, model_dict: dict, exploration_params=None, return_details: bool = True):
        """Send roll-out requests to remote actors.

        Args:
            model_dict (dict): Models the remote actors .
            exploration_params: Exploration parameters.
            return_details (bool): If True, return experiences as well as performance metrics provided by the env.

        Returns:
            Performance and per-agent experiences from the remote actor.
        """
        payloads = [(peer, {PayloadKey.MODEL: model_dict,
                            PayloadKey.EXPLORATION_PARAMS: exploration_params,
                            PayloadKey.RETURN_DETAILS: return_details})
                    for peer in self._proxy.peers_name["actor"]]
        # TODO: double check when ack enable
        replies = self._proxy.scatter(
            tag=MessageTag.ROLLOUT,
            session_type=SessionType.TASK,
            destination_payload_list=payloads
        )

        performance = [(msg.source, msg.payload[PayloadKey.PERFORMANCE]) for msg in replies]
        details_by_source = {msg.source: msg.payload[PayloadKey.DETAILS] for msg in replies}
        details = self._experience_collecting_func(details_by_source) if return_details else None

        return performance, details


class SEEDLearner(AbsDistLearner):
    """A simple implementation of ``AbsLearner``.

    Args:
        agent_manager (AbsAgentManager): An AgentManager instance that manages all agents.
        scheduler (AbsScheduler): A scheduler responsible for iterating over episodes and generating exploration
            parameters if necessary.
    """
    def __init__(self, agent_manager, scheduler, experience_collecting_func, **proxy_params):
        super().__init__(agent_manager, scheduler, experience_collecting_func, **proxy_params)
        self._num_actors = len(self._proxy.peers_name["actor"])
        self._registry_table.register_event_handler(
            f"actor:{MessageTag.CHOOSE_ACTION.value}:{self._num_actors}", self._get_action
        )
        self._registry_table.register_event_handler(
            f"actor:{MessageTag.UPDATE.value}:{self._num_actors}", self._collect)
        self._performances = {}
        self._details = {}

    def learn(self):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        for exploration_params in self._scheduler:
            self._details.clear()
            self._performances.clear()
            # load exploration parameters:
            if exploration_params is not None:
                self._agent_manager.update_exploration_params(exploration_params)
            self._sample()
            self._serve()
            for perf in self._performances.values():
                self._scheduler.record_performance(perf)
            exp_by_agent = self._experience_collecting_func(self._details)
            self._agent_manager.train(exp_by_agent)

    def test(self):
        """Test policy performance."""
        self._sample()
        self._serve()

    def _sample(self, return_details: bool = True):
        """Send roll-out requests to remote actors.

        Args:
            return_details (bool): If True, return experiences as well as performance metrics provided by the env.

        Returns:
            Performance and per-agent experiences from the remote actor.
        """
        payloads = [(peer, {PayloadKey.RETURN_DETAILS: return_details})
                    for peer in self._proxy.peers_name["actor"]]
        self._proxy.iscatter(
            tag=MessageTag.ROLLOUT,
            session_type=SessionType.TASK,
            destination_payload_list=payloads
        )

    def _serve(self):
        for msg in self._proxy.receive():
            self._registry_table.push(msg)
            for handler_fn, cached_messages in self._registry_table.get():
                handler_fn(cached_messages)
            if len(self._performances) == self._num_actors:
                break

    def _get_action(self, messages: list):
        state_batch = np.vstack([msg.payload[PayloadKey.STATE] for msg in messages])
        agent_id = messages[0].payload[PayloadKey.AGENT_ID]
        model_action_batch = self._agent_manager[agent_id].choose_action(state_batch)
        for msg, model_action in zip(messages, model_action_batch):
            self._proxy.reply(received_message=msg, tag=MessageTag.ACTION, payload={PayloadKey.ACTION: model_action})

    def _collect(self, messages: list):
        for msg in messages:
            self._performances[msg.source] = msg.payload[PayloadKey.PERFORMANCE]
            self._details[msg.source] = msg.payload[PayloadKey.DETAILS]
