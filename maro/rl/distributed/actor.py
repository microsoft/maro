# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from abc import ABC
from typing import Union

from maro.communication import Proxy
from maro.rl.agent_manager import AgentManager
from maro.simulator import Env
from maro.utils import InternalLogger

from .agent_manager_proxy import AgentManagerProxy
from .common import MessageTag, PayloadKey, TerminateEpisode


class Actor(ABC):
    """Abstract actor class.

    Args:
        env: An environment instance.
        agent_manager: An ``AgentManager`` or ``AgentManagerProxy`` instance responsible for interacting with the
            environment.
        proxy_params: Parameters required for instantiating an internal proxy for communication.
    """
    def __init__(
        self,
        env: Env,
        agent_manager: Union[AgentManager, AgentManagerProxy],
        proxy: Proxy
    ):
        self._env = env
        self._agent_manager = agent_manager
        self._proxy = proxy
        self._logger = InternalLogger(self._proxy.component_name)
        self._expected_ep = 0

    def run(self):
        """Entry point method.

        This enters the actor into an infinite loop of listening to requests and handling them according to the
        register table. In this case, the only type of requests the actor needs to handle is roll-out requests.
        """
        for msg in self._proxy.receive():
            if msg.tag == MessageTag.EXIT:
                self.exit()
            elif msg.tag == MessageTag.ROLLOUT:
                self._roll_out(msg)

    def _roll_out(self, message):
        """Perform one episode of roll-out and send performance and experiences back to the learner.

        Args:
            message: Message containing roll-out parameters and options.
        """
        self._env.reset()
        ep = message.payload[PayloadKey.EPISODE]
        if isinstance(self._agent_manager, AgentManager):
            model_dict = message.payload.get(PayloadKey.MODEL, None)
            if model_dict is not None:
                self._agent_manager.load_models(model_dict)
            exploration_params = message.payload.get(PayloadKey.EXPLORATION_PARAMS, None)
            if exploration_params is not None:
                self._agent_manager.set_exploration_params(exploration_params)
        else:
            self._agent_manager.reset(ep)

        self._logger.info(f"Rolling out for ep-{ep}...")
        metrics, decision_event, is_done = self._env.step(None)
        while not is_done:
            action = self._agent_manager.choose_action(decision_event, self._env.snapshot_list)
            # Received action is an TERMINATE_EPISODE command from learner
            if isinstance(action, TerminateEpisode):
                self._logger.info(f"Roll-out aborted at time step {self._agent_manager.time_step}.")
                return

            metrics, decision_event, is_done = self._env.step(action)
            if action:
                self._agent_manager.on_env_feedback(metrics)
            else:
                self._logger.info(
                    f"Failed to receive an action for time step {self._agent_manager.time_step}, "
                    f"proceed with NULL action."
                )

        payload = {
            PayloadKey.EPISODE: ep,
            PayloadKey.PERFORMANCE: self._env.metrics,
            PayloadKey.EXPERIENCES:
                self._agent_manager.post_process(self._env.snapshot_list)
                if message.payload[PayloadKey.IS_TRAINING] else None
        }

        # If the agent manager is an AgentManagerProxy instance (SEED architecture), the actor needs
        # to tell the learner the ID of the agent manager so that the learner can send termination
        # signals to the agent managers of unfinished actors.
        if isinstance(self._agent_manager, AgentManagerProxy):
            payload[PayloadKey.AGENT_MANAGER_ID] = self._agent_manager.agents.component_name

        self._proxy.reply(message, tag=MessageTag.FINISHED, payload=payload)
        self._logger.info(f"Roll-out finished for ep-{ep}")

    def exit(self):
        self._logger.info("Exiting...")
        sys.exit(0)
