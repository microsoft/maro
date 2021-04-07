# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Union

from maro.communication import Message, Proxy, SessionType
from maro.utils import InternalLogger

from .message_enums import MsgTag, MsgKey


class ActorManager(object):
    """Learner class for distributed training.

    Args:
        agent (Union[AbsAgent, MultiAgentWrapper]): Learning agents.
        scheduler (Scheduler): .
        num_actors (int): Expected number of actors in the group identified by ``group_name``.
        group_name (str): Identifier of the group to which the actor belongs. It must be the same group name
            assigned to the actors (and roll-out clients, if any).
        proxy_options (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
        update_trigger (str): Number or percentage of ``MsgTag.ROLLOUT_DONE`` messages required to trigger
            learner updates, i.e., model training.
    """
    def __init__(
        self,
        num_actors: int,
        group_name: str,
        proxy_options: dict = None,
        log_env_metrics: bool = False
    ):
        super().__init__()
        peers = {"actor": num_actors}
        if proxy_options is None:
            proxy_options = {}
        self._proxy = Proxy(group_name, "actor_manager", peers, **proxy_options)
        self._actors = self._proxy.peers["actor"]  # remote actor ID's
        self.total_experiences_collected = 0
        self.total_env_steps = 0
        self._log_env_metrics = log_env_metrics
        self._logger = InternalLogger("ACTOR_MANAGER")

    def collect(
        self,
        rollout_index: int,
        segment_index: int,
        num_steps: int,
        models: dict = None,
        exploration_params=None,
        required_actor_finishes: int = None,
        discard_stale_experiences: bool = True
    ):
        """Collect experiences from actors."""
        if required_actor_finishes is None:
            required_actor_finishes = len(self._actors)

        msg_body = {
            MsgKey.ROLLOUT_INDEX: rollout_index,
            MsgKey.SEGMENT_INDEX: segment_index,
            MsgKey.NUM_STEPS: num_steps,
            MsgKey.MODEL: models
        }

        if exploration_params:
            msg_body[MsgKey.EXPLORATION_PARAMS] = exploration_params

        if self._log_env_metrics:
            self._logger.info(f"EPISODE-{rollout_index}, SEGMENT-{segment_index}: ")
            self._logger.info(f"exploration_params: {exploration_params}")

        self._proxy.ibroadcast("actor", MsgTag.ROLLOUT, SessionType.TASK, body=msg_body)
        self._logger.info(f"Sent roll-out requests for ep-{rollout_index}, segment-{segment_index}")

        # Receive roll-out results from remote actors
        num_finishes = 0
        for msg in self._proxy.receive():
            if msg.body[MsgKey.ROLLOUT_INDEX] != rollout_index:
                self._logger.info(
                    f"Ignore a message of type {msg.tag} with ep {msg.body[MsgKey.ROLLOUT_INDEX]} "
                    f"(expected {rollout_index})"
                )
                continue

            # log roll-out summary
            if self._log_env_metrics:
                env_metrics = msg.body[MsgKey.METRICS]
                self._logger.info(f"env_metrics: {env_metrics}")

            if msg.body[MsgKey.SEGMENT_INDEX] != segment_index:
                if not discard_stale_experiences:
                    experiences = msg.body[MsgKey.EXPERIENCES]
                    self.total_experiences_collected += msg.body[MsgKey.NUM_EXPERIENCES]
                    self.total_env_steps += num_steps
                    yield msg.body[MsgKey.EXPERIENCES], msg.body[MsgKey.ENV_END]
            else:
                self.total_experiences_collected += msg.body[MsgKey.NUM_EXPERIENCES]
                self.total_env_steps += num_steps
                yield msg.body[MsgKey.EXPERIENCES], msg.body[MsgKey.ENV_END]
                num_finishes += 1

            if num_finishes == required_actor_finishes:
                break

    def exit(self):
        """Tell the remote actors to exit."""
        self._proxy.ibroadcast("actor", MsgTag.EXIT, SessionType.NOTIFICATION)
        self._logger.info("Exiting...")
