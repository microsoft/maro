# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from abc import ABC
from typing import Union

from maro.communication import Proxy
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

    def launch(self):
        """Entry point method.

        This enters the actor into an infinite loop of listening to requests and handling them according to the
        register table. In this case, the only type of requests the actor needs to handle is roll-out requests.
        """
        while True:
            msg = self._proxy.receive(is_continuous=False)
            if msg.tag == MessageTag.ROLLOUT:
                self._roll_out(msg)
            elif msg.tag == MessageTag.EXIT:
                sys.exit(0)

    def _roll_out(self, message):
        """Perform one episode of roll-out and send performance and experiences back to the learner.

        Args:
            message: Message containing roll-out parameters and options.
        """
        self._env.reset()
        if isinstance(self._executor, AbsAgentManager):
            model_dict = message.payload.get(PayloadKey.MODEL, None)
            if model_dict is not None:
                self._executor.load_models(model_dict)
            exploration_params = message.payload.get(PayloadKey.EXPLORATION_PARAMS, None)
            if exploration_params is not None:
                self._executor.update_exploration_params(exploration_params)
        else:
            self._executor.set_ep(int(message.session_id.split("_")[-1]))

        metrics, decision_event, is_done = self._env.step(None)
        while not is_done:
            action = self._executor.choose_action(decision_event, self._env.snapshot_list)
            metrics, decision_event, is_done = self._env.step(action)
            # Force reset
            if action == -1:
                return
            if action:
                self._executor.on_env_feedback(metrics)

        is_training = message.payload[PayloadKey.IS_TRAINING]
        self._proxy.reply(
            received_message=message,
            tag=MessageTag.UPDATE,
            payload={
                PayloadKey.EPISODE: message.payload[PayloadKey.EPISODE],
                PayloadKey.PERFORMANCE: self._env.metrics,
                PayloadKey.EXPERIENCES: self._executor.post_process(self._env.snapshot_list) if is_training else None
            }
        )
