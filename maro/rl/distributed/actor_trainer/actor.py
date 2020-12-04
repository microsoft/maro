# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC
from typing import Union

from maro.communication import Proxy, SessionMessage
from maro.communication.registry_table import RegisterTable
from maro.rl.agent.abs_agent_manager import AbsAgentManager
from maro.rl.scheduling.scheduler import Scheduler
from maro.simulator import Env

from ..common import ActorTrainerComponent, MessageTag, PayloadKey
from ..executor import Executor


class AutoActor(ABC):
    """Abstract actor class.

    Args:
        env: An environment instance.
        proxy_params: Parameters required for instantiating an internal proxy for communication.
    """
    def __init__(self, env: Env, executor: Union[AbsAgentManager, Executor], scheduler: Scheduler, **proxy_params):
        self._env = env
        self._executor = executor
        self._scheduler = scheduler
        self._proxy = Proxy(component_type=ActorTrainerComponent.ACTOR.value, **proxy_params)
        if isinstance(self._executor, Executor):
            self._executor.load_proxy(self._proxy)
        self._registry_table = RegisterTable(self._proxy.peers_name)

    def run(self, is_training: bool = True):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        for exploration_params in self._scheduler:
            self._env.reset()
            if exploration_params is not None:
                self._update_exploration_params(exploration_params)
            metrics, decision_event, is_done = self._env.step(None)
            while not is_done:
                action = self._executor.choose_action(decision_event, self._env.snapshot_list)
                metrics, decision_event, is_done = self._env.step(action)
                self._executor.on_env_feedback(metrics)

            if is_training:
                self._scheduler.record_performance(self._env.metrics)
                experiences = self._executor.post_process(self._env.snapshot_list)
                reply = self._request_update(experiences)
                if isinstance(self._executor, AbsAgentManager):
                    self._executor.load_models(reply[0].payload[PayloadKey.MODEL])

    def _update_exploration_params(self, exploration_params):
        # load exploration parameters
        if isinstance(self._executor, AbsAgentManager):
            if exploration_params is not None:
                self._executor.update_exploration_params(exploration_params)
        else:
            self._proxy.send(
                SessionMessage(
                    tag=MessageTag.EXPLORATION_PARAMS,
                    source=self._proxy.component_name,
                    destination=self._proxy.peers_name[ActorTrainerComponent.TRAINER.value][0],
                    session_id=self._get_update_session_id(),
                    payload={
                        PayloadKey.ACTOR_ID: self._proxy.component_name,
                        PayloadKey.EXPLORATION_PARAMS: exploration_params
                    },
                )
            )

    def _get_update_session_id(self):
        return ".".join([
            f"ep_{self._scheduler.current_ep}",
            ActorTrainerComponent.ACTOR.value,
            ActorTrainerComponent.TRAINER.value
            ])

    def _request_update(self, experiences):
        return self._proxy.send(
            SessionMessage(
                tag=MessageTag.UPDATE,
                source=self._proxy.component_name,
                destination=self._proxy.peers_name[ActorTrainerComponent.TRAINER.value][0],
                session_id=self._get_update_session_id(),
                payload={PayloadKey.EXPERIENCES: experiences},
            )
        )
