# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from typing import Callable

from maro.communication import Proxy, RegisterTable, SessionType
from maro.rl.agent.simple_agent_manager import SimpleAgentManager
from maro.rl.dist_topologies.common import MessageTag, PayloadKey
from maro.rl.scheduling.scheduler import Scheduler

from maro.rl.learner.abs_learner import AbsLearner


class DistLearner(AbsLearner):
    """A simple implementation of ``AbsLearner``.

    Args:
        agent_manager (AbsAgentManager): An AgentManager instance that manages all agents.
        scheduler (AbsScheduler): A scheduler responsible for iterating over episodes and generating exploration
            parameters if necessary.
    """
    def __init__(
        self,
        agent_manager: SimpleAgentManager,
        scheduler: Scheduler,
        proxy_params: dict,
        experience_collecting_func: Callable
    ):
        super().__init__()
        self._agent_manager = agent_manager
        self._scheduler = scheduler
        self._proxy = Proxy(component_type="learner", **proxy_params)
        self._num_actors = len(self._proxy.peers_name["actor"])
        self._registry_table = RegisterTable(self._proxy.peers_name)
        self._registry_table.register_event_handler("actor:choose_action:1", self._get_action)
        self._registry_table.register_event_handler("actor:result:1", self._collect)
        self._experience_collecting_func = experience_collecting_func
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
            self._remote_rollout()
            self._serve()
            for perf in self._performances.values():
                self._scheduler.record_performance(perf)
            exp_by_agent = self._experience_collecting_func(self._details)
            self._agent_manager.train(exp_by_agent)

    def test(self):
        """Test policy performance."""
        self._remote_rollout()
        self._serve()

    def exit(self, code: int = 0):
        """Tell the remote actor to exit."""
        self._remote_rollout(done=True)
        sys.exit(code)

    def load_models(self, dir_path: str):
        self._agent_manager.load_models_from_files(dir_path)

    def dump_models(self, dir_path: str):
        self._agent_manager.dump_models_to_files(dir_path)

    def _remote_rollout(self, done: bool = False, return_details: bool = True):
        if done:
            self._proxy.ibroadcast(
                component_type="actor",
                tag=MessageTag.ROLLOUT,
                session_type=SessionType.NOTIFICATION,
                payload={PayloadKey.DONE: True}
            )
            return None, None

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
            triggered_events = self._registry_table.get()
            for handler_fn, cached_messages in triggered_events:
                handler_fn(cached_messages)
            if len(self._performances) == self._num_actors:
                break

    def _get_action(self, message):
        state, agent_id = message.payload[PayloadKey.STATE], message.payload[PayloadKey.AGENT_ID]
        model_action = self._agent_manager[agent_id].choose_action(state)
        self._proxy.reply(
            received_message=message,
            tag=MessageTag.ACTION,
            payload={PayloadKey.ACTION: model_action}
        )

    def _collect(self, message):
        self._performances[message.source] = message.payload[PayloadKey.PERFORMANCE]
        self._details[message.source] = message.payload[PayloadKey.DETAILS]
