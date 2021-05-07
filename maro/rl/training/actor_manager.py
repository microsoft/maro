# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from os import getcwd
from random import choices
from typing import Union

from maro.communication import Proxy, SessionType
from maro.utils import Logger

from .message_enums import MsgTag, MsgKey


class ActorManager(object):
    """Learner class for distributed training.

    Args:
        agent (Union[AbsPolicy, MultiAgentWrapper]): Learning agents.
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
        required_finishes: int = None, 
        log_env_metrics: bool = False,
        log_dir: str = getcwd()
    ):
        super().__init__()
        self.num_actors = num_actors
        peers = {"actor": num_actors}
        if proxy_options is None:
            proxy_options = {}
        self._proxy = Proxy(group_name, "actor_manager", peers, **proxy_options)
        self._actors = self._proxy.peers["actor"]  # remote actor ID's

        if required_finishes and required_finishes > self.num_actors:
            raise ValueError("required_finishes cannot exceed the number of available actors")

        if required_finishes is None:
            required_finishes = self.num_actors
            self._logger.info(f"Required number of actor finishes is set to {required_finishes}")

        self.required_finishes = required_finishes
        self.total_experiences_collected = 0
        self.total_env_steps = 0
        self.total_reward = defaultdict(float)
        self._log_env_metrics = log_env_metrics
        self._logger = Logger("ACTOR_MANAGER", dump_folder=log_dir)

    def collect(
        self,
        episode_index: int,
        segment_index: int,
        num_steps: int,
        policy_dict: dict = None,
        exploration=None,
        discard_stale_experiences: bool = True,
        return_env_metrics: bool = False
    ):
        """Collect experiences from actors."""
        msg_body = {
            MsgKey.EPISODE_INDEX: episode_index,
            MsgKey.SEGMENT_INDEX: segment_index,
            MsgKey.NUM_STEPS: num_steps,
            MsgKey.POLICY: policy_dict,
            MsgKey.RETURN_ENV_METRICS: return_env_metrics
        }

        if self._log_env_metrics:
            self._logger.info(f"EPISODE-{episode_index}, SEGMENT-{segment_index}: ")
            if exploration_params:
                self._logger.info(f"exploration_params: {exploration_params}")

        self._proxy.ibroadcast("actor", MsgTag.COLLECT, SessionType.TASK, body=msg_body)
        self._logger.info(f"Sent collect requests for ep-{episode_index}, segment-{segment_index}")

        # Receive roll-out results from remote actors
        num_finishes = 0
        for msg in self._proxy.receive():
            if msg.tag != MsgTag.COLLECT_DONE or msg.body[MsgKey.EPISODE_INDEX] != episode_index:
                self._logger.info(
                    f"Ignore a message of type {msg.tag} with roll-out index {msg.body[MsgKey.EPISODE_INDEX]} "
                    f"(expected message type {MsgTag.COLLECT} and roll-out index {episode_index})"
                )
                continue

            # log roll-out summary
            if self._log_env_metrics:
                env_metrics = msg.body[MsgKey.METRICS]
                self._logger.info(f"env_metrics: {env_metrics}")

            if msg.body[MsgKey.SEGMENT_INDEX] == segment_index or not discard_stale_experiences:
                experiences_by_agent = msg.body[MsgKey.EXPERIENCES]
                self.total_experiences_collected += sum(len(exp) for exp in experiences_by_agent.values())
                self.total_env_steps += msg.body[MsgKey.NUM_STEPS]
                is_env_end = msg.body[MsgKey.ENV_END]
                if is_env_end:
                    self._logger.info(f"total rewards: {msg.body[MsgKey.TOTAL_REWARD]}")
                yield experiences_by_agent, is_env_end

            if msg.body[MsgKey.SEGMENT_INDEX] == segment_index:
                num_finishes += 1
                if num_finishes == self.required_finishes:
                    break

    def evaluate(self, episode_index: int, policy_dict: dict, num_actors: int):
        """Collect experiences from actors."""
        msg_body = {
            MsgKey.EPISODE_INDEX: episode_index,
            MsgKey.POLICY: policy_dict,
            MsgKey.RETURN_ENV_METRICS: True
        }

        actors = choices(self._actors, k=num_actors)
        self._proxy.iscatter(MsgTag.EVAL, SessionType.TASK, [(actor_id, msg_body) for actor_id in actors])
        self._logger.info(f"Sent evaluation requests to {actors}")

        # Receive roll-out results from remote actors
        num_finishes = 0
        for msg in self._proxy.receive():
            if msg.tag != MsgTag.EVAL_DONE or msg.body[MsgKey.EPISODE_INDEX] != episode_index:
                self._logger.info(
                    f"Ignore a message of type {msg.tag} with episode index {msg.body[MsgKey.EPISODE_INDEX]} "
                    f"(expected message type {MsgTag.EVAL} and episode index {episode_index})"
                )
                continue

            # log roll-out summary
            env_metrics = msg.body[MsgKey.METRICS]
            self._logger.info(f"env metrics for evaluation episode {episode_index}: {env_metrics}")

            if msg.body[MsgKey.EPISODE_INDEX] == episode_index:
                num_finishes += 1
                if num_finishes == num_actors:
                    break

    def exit(self):
        """Tell the remote actors to exit."""
        self._proxy.ibroadcast("actor", MsgTag.EXIT, SessionType.NOTIFICATION)
        self._logger.info("Exiting...")
