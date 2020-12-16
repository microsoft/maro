# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from abc import ABC
from typing import Union

from maro.communication import Message, Proxy
from maro.rl.agent.abs_agent_manager import AbsAgentManager
from maro.simulator import Env
from maro.utils import DummyLogger, Logger

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
    def __init__(
        self,
        env: Env,
        executor: Union[AbsAgentManager, Executor],
        logger: Logger = DummyLogger(),
        **proxy_params
    ):
        self._env = env
        self._executor = executor
        self._proxy = Proxy(component_type=ACTOR, **proxy_params)
        if isinstance(self._executor, Executor):
            self._executor.load_proxy(self._proxy)
        self._logger = logger
        self._expected_ep = 0

    def run(self):
        """Entry point method.

        This enters the actor into an infinite loop of listening to requests and handling them according to the
        register table. In this case, the only type of requests the actor needs to handle is roll-out requests.
        """
        while True:
            message = self._proxy.receive_by_id(
                (MessageTag.ROLLOUT, "*"), stop_condition=lambda msg: msg.tag == MessageTag.EXIT
            )
            if isinstance(message, Message):
                sys.exit(0)

            message = message[0]
            if message.session_id == "test":
                ret = self._roll_out(message)
                if ret and ret.tag == MessageTag.EXIT:
                    sys.exit(0)
            else:
                ep = int(message.session_id.split("-")[-1])
                if ep < self._expected_ep:
                    self._logger.info(
                        f"{self._proxy.component_name} expects roll-out requests for episode >= {self._expected_ep}. "
                        f"Current request ({message.session_id}) ignored."
                    )
                    continue
                self._expected_ep = ep
                ret = self._roll_out(message)
                if not ret:
                    self._expected_ep += 1
                elif ret.tag == MessageTag.EXIT:
                    self.exit()

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
            self._executor.set_ep("test" if message.session_id == "test" else int(message.session_id.split("-")[-1]))

        self._logger.info(f"{self._proxy.component_name} performing roll-out for {message.session_id}")
        metrics, decision_event, is_done = self._env.step(None)
        while not is_done:
            action = self._executor.choose_action(decision_event, self._env.snapshot_list)
            if action is None:
                self._logger.debug(
                    f"{self._proxy.component_name} failed to receive an action before timeout, "
                    f"proceeding with NULL action."
                )
            # Reset or exit
            if isinstance(action, Message):
                self._logger.info(
                    f"{self._proxy.component_name} received a message with tag {message.tag} and "
                    f"session {message.session_id}. Roll-out aborted.")
                return action

            metrics, decision_event, is_done = self._env.step(action)
            if action:
                self._executor.on_env_feedback(metrics)

        self._proxy.reply(
            received_message=message,
            tag=MessageTag.FINISHED,
            payload={
                PayloadKey.PERFORMANCE: self._env.metrics,
                PayloadKey.EXPERIENCES:
                    None if message.session_id == "test" else self._executor.post_process(self._env.snapshot_list)
            }
        )
        self._logger.info(f"{self._proxy.component_name} finished roll-out for {message.session_id}")

    def exit(self):
        self._logger.info(f"{self._proxy.component_name} exiting...")
        sys.exit(0)
