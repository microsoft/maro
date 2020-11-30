# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from typing import Callable

import numpy as np

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
        experience_collecting_func: Callable,

    ):
        super().__init__()
        self._agent_manager = agent_manager
        self._scheduler = scheduler
        self._proxy = Proxy(component_type="learner", **proxy_params)
        self._num_actors = len(self._proxy.peers_name["actor"])
        self._registry_table = RegisterTable(self._proxy.peers_name)
        self._registry_table.register_event_handler(f"actor:choose_action:{self._num_actors}", self._get_action)
        self._registry_table.register_event_handler(f"actor:update:{self._num_actors}", self._collect)
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
            self._rollout()
            self._serve()
            for perf in self._performances.values():
                self._scheduler.record_performance(perf)
            exp_by_agent = self._experience_collecting_func(self._details)
            self._agent_manager.train(exp_by_agent)

    def test(self):
        """Test policy performance."""
        self._rollout()
        self._serve()

    def exit(self, code: int = 0):
        """Tell the remote actor to exit."""
        self._rollout(done=True)
        sys.exit(code)

    def load_models(self, dir_path: str):
        self._agent_manager.load_models_from_files(dir_path)

    def dump_models(self, dir_path: str):
        self._agent_manager.dump_models_to_files(dir_path)

    def _rollout(self, done: bool = False, return_details: bool = True):
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
            for handler_fn, cached_messages in self._registry_table.get():
                handler_fn(cached_messages)
            if len(self._performances) == self._num_actors:
                break

    def _get_action(self, messages: list):
        state_batch = np.vstack([msg.payload[PayloadKey.STATE] for msg in messages])
        agent_id = messages[0].payload[PayloadKey.AGENT_ID]
        model_action_batch = self._agent_manager[agent_id].choose_action(state_batch)
        self._proxy.iscatter(
            tag=MessageTag.ACTION,
            session_type=SessionType.NOTIFICATION,
            destination_payload_list=[
                (msg.source, model_action for msg, model_action in zip(messages, model_action_batch))
            ]
        )

    def _collect(self, messages: list):
        for msg in messages:
            self._performances[msg.source] = msg.payload[PayloadKey.PERFORMANCE]
            self._details[msg.source] = msg.payload[PayloadKey.DETAILS]
