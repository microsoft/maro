# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os import getcwd
from typing import List

from maro.communication import Message, Proxy, RegisterTable, SessionType
from maro.utils import Logger

from .message_enums import MessageTag, PayloadKey


class ActorProxy(object):
    """Actor proxy that manages a set of remote actors.

    Args:
        group_name (str): Identifier of the group to which the actor belongs. It must be the same group name
            assigned to the actors (and roll-out clients, if any).
        num_actors (int): Expected number of actors in the group identified by ``group_name``.
        update_trigger (str): Number or percentage of ``MessageTag.FINISHED`` messages required to trigger
            learner updates, i.e., model training.
        proxy_options (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
    """
    def __init__(
        self,
        group_name: str,
        num_actors: int,
        update_trigger: str = None,
        proxy_options: dict = None,
        log_dir: str = getcwd()
    ):
        self.agent = None
        peers = {"actor": num_actors}
        if proxy_options is None:
            proxy_options = {}
        self._proxy = Proxy(group_name, "learner", peers, **proxy_options)
        self._actors = self._proxy.peers_name["actor"]  # remote actor ID's
        self._registry_table = RegisterTable(self._proxy.peers_name)
        if update_trigger is None:
            update_trigger = len(self._actors)
        self._registry_table.register_event_handler(
            f"actor:{MessageTag.FINISHED.value}:{update_trigger}", self._on_rollout_finish
        )
        self.logger = Logger("ACTOR_PROXY", dump_folder=log_dir)

    def roll_out(self, index: int, training: bool = True, model_by_agent: dict = None, exploration_params=None):
        """Collect roll-out data from remote actors.

        Args:
            index (int): Index of roll-out requests.
            training (bool): If true, the roll-out request is for training purposes.
            model_by_agent (dict): Models to be broadcast to remote actors for inference. Defaults to None.
            exploration_params: Exploration parameters to be used by the remote roll-out actors. Defaults to None.
        """
        payload = {
            PayloadKey.ROLLOUT_INDEX: index,
            PayloadKey.TRAINING: training,
            PayloadKey.MODEL: model_by_agent,
            PayloadKey.EXPLORATION_PARAMS: exploration_params
        }
        self._proxy.iscatter(MessageTag.ROLLOUT, SessionType.TASK, [(actor, payload) for actor in self._actors])
        self.logger.info(f"Sent roll-out requests to {self._actors} for ep-{index}")

        # Receive roll-out results from remote actors
        for msg in self._proxy.receive():
            if msg.payload[PayloadKey.ROLLOUT_INDEX] != index:
                self.logger.info(
                    f"Ignore a message of type {msg.tag} with ep {msg.payload[PayloadKey.ROLLOUT_INDEX]} "
                    f"(expected {index} or greater)"
                )
                continue
            if msg.tag == MessageTag.FINISHED:
                # If enough update messages have been received, call update() and break out of the loop to start
                # the next episode.
                result = self._registry_table.push(msg)
                if result:
                    env_metrics, details = result[0]
                    break

        return env_metrics, details

    def _on_rollout_finish(self, messages: List[Message]):
        metrics = {msg.source: msg.payload[PayloadKey.METRICS] for msg in messages}
        details = {msg.source: msg.payload[PayloadKey.DETAILS] for msg in messages}
        return metrics, details

    def terminate(self):
        """Tell the remote actors to exit."""
        self._proxy.ibroadcast(
            component_type="actor", tag=MessageTag.EXIT, session_type=SessionType.NOTIFICATION
        )
        self.logger.info("Exiting...")
        self._proxy.close()
