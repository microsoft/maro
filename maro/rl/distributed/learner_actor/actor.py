# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from abc import ABC
from typing import Union

from maro.communication import Proxy
from maro.communication.registry_table import RegisterTable
from maro.rl.agent.abs_agent_manager import AbsAgentManager
from maro.simulator import Env

from .common import Component, MessageTag, PayloadKey
from ..executor import Executor

ACTOR = Component.ACTOR.value
LEARNER = Component.LEARNER.value


class Actor(ABC):
    """Abstract actor class.

    Args:
        env: An environment instance.
        executor: An ``Executor`` of ``AbsAgentManager`` instance responsible for interacting with the environment.
        proxy_params: Parameters required for instantiating an internal proxy for communication.
    """
    def __init__(self, env: Env, executor: Union[AbsAgentManager, Executor], **proxy_params):
        self._env = env
        self._executor = executor
        self._proxy = Proxy(component_type=ACTOR, **proxy_params)
        if isinstance(self._executor, Executor):
            self._executor.load_proxy(self._proxy)
        self._registry_table = RegisterTable(self._proxy.peers_name)
        self._registry_table.register_event_handler(f"{LEARNER}:{MessageTag.ROLLOUT.value}:1", self.on_rollout_request)
        self._registry_table.register_event_handler(f"{LEARNER}:{MessageTag.EXIT.value}:1", self.exit)
        self._current_ep = None

    def launch(self):
        """Entry point method.

        This enters the actor into an infinite loop of listening to requests and handling them according to the
        register table. In this case, the only type of requests the actor needs to handle is roll-out requests.
        """
        for msg in self._proxy.receive():
            self._registry_table.push(msg)
            triggered_events = self._registry_table.get()
            for handler_fn, cached_messages in triggered_events:
                handler_fn(cached_messages)

    def on_rollout_request(self, message):
        """Perform local roll-out and send the results back to the request sender.

        Args:
            message: Message containing roll-out parameters and options.
        """
        performance, experiences = self._roll_out(
            model_dict=message.payload.get(PayloadKey.MODEL, None),
            exploration_params=message.payload.get(PayloadKey.EXPLORATION_PARAMS, None),
            return_experiences=message.payload[PayloadKey.RETURN_EXPERIENCES]
        )

        self._proxy.reply(
            received_message=message,
            tag=MessageTag.UPDATE,
            payload={
                PayloadKey.EPISODE: message.payload[PayloadKey.EPISODE],
                PayloadKey.PERFORMANCE: performance,
                PayloadKey.EXPERIENCES: experiences
            }
        )

    def _roll_out(self, model_dict: dict = None, exploration_params=None, return_experiences: bool = True):
        """Perform one episode of roll-out and return performance and experiences.

        Args:
            model_dict (dict): If not None, the agents will load the models from model_dict and use these models
                to perform roll-out.
            exploration_params: Exploration parameters.
            return_experiences (bool): If True, return experiences as well as performance metrics provided by the env.

        Returns:
            Performance and relevant experiences from the episode (e.g., experiences).
        """
        self._env.reset()

        if isinstance(self._executor, AbsAgentManager):
            # load models
            if model_dict is not None:
                self._executor.load_models(model_dict)

            # load exploration parameters:
            if exploration_params is not None:
                self._executor.update_exploration_params(exploration_params)

        metrics, decision_event, is_done = self._env.step(None)
        while not is_done:
            action = self._executor.choose_action(decision_event, self._env.snapshot_list)
            if action:
                metrics, decision_event, is_done = self._env.step(action)
                self._executor.on_env_feedback(metrics)

        experiences = self._executor.post_process(self._env.snapshot_list) if return_experiences else None

        return self._env.metrics, experiences

    def exit(self):
        sys.exit(0)
