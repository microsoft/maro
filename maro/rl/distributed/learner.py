# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Union

from maro.communication import Message, Proxy, SessionType
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.scheduling import Scheduler
from maro.utils import InternalLogger

from .message_enums import MsgTag, MsgKey


class DistLearner(object):
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
        agent: Union[AbsAgent, MultiAgentWrapper],
        scheduler: Scheduler,
        num_actors: int,
        group_name: str,
        proxy_options: dict = None,
        agent_update_interval: int = -1,
        min_actor_finishes: str = None,
        ignore_stale_experiences: bool = True
    ):
        super().__init__()
        self.agent = MultiAgentWrapper(agent) if isinstance(agent, AbsAgent) else agent
        self.scheduler = scheduler
        peers = {"actor": num_actors}
        if proxy_options is None:
            proxy_options = {}
        self._proxy = Proxy(group_name, "learner", peers, **proxy_options)
        self.actors = self._proxy.peers["actor"]  # remote actor ID's
        if min_actor_finishes is None:
            self.min_actor_finishes = len(self.actors)
        self.agent_update_interval = agent_update_interval
        self.ignore_stale_experiences = ignore_stale_experiences
        self._logger = InternalLogger("LEARNER")

    def run(self):
        """Main learning loop"""
        for exploration_params in self.scheduler:
            updated_agents = self.agent.names
            rollout_index, segment_index, num_episode_finishes = self.scheduler.iter, 0, 0
            while num_episode_finishes < self.min_actor_finishes:
                msg_body = {
                    MsgKey.ROLLOUT_INDEX: rollout_index,
                    MsgKey.SEGMENT_INDEX: segment_index,
                    MsgKey.NUM_STEPS: self.agent_update_interval,
                    MsgKey.MODEL: self.agent.dump_model(agent_ids=updated_agents)
                }
                if segment_index == 0 and exploration_params:
                    msg_body[MsgKey.EXPLORATION_PARAMS] = exploration_params
                self._proxy.ibroadcast("actor", MsgTag.ROLLOUT, SessionType.TASK, body=msg_body)
                self._logger.info(f"Sent roll-out requests for ep-{rollout_index}, segment-{segment_index}")

                # Receive roll-out results from remote actors
                updated_agents, num_segment_finishes = set(), 0
                for msg in self._proxy.receive():
                    if msg.body[MsgKey.ROLLOUT_INDEX] != rollout_index:
                        self._logger.info(
                            f"Ignore a message of type {msg.tag} with ep {msg.body[MsgKey.ROLLOUT_INDEX]} "
                            f"(expected {index})"
                        )
                        continue
                    
                    env_metrics = msg.body[MsgKey.METRICS]
                    self._logger.info(
                        f"ep-{rollout_index}, segment-{segment_index}: {env_metrics} ({exploration_params})"
                    )
                    if msg.body[MsgKey.SEGMENT_INDEX] == segment_index or not self.ignore_stale_experiences:
                        updated_agents.update(self.agent.update(msg.body[MsgKey.EXPERIENCES]))
                        self._logger.info(f"Learning finished for agent {updated_agents}")
                    if msg.body[MsgKey.END_OF_EPISODE]:
                        num_episode_finishes += 1
                    num_segment_finishes += 1
                    if num_segment_finishes == self.min_actor_finishes:
                        break

                segment_index += 1

    def terminate(self):
        """Tell the remote actors to exit."""
        self._proxy.ibroadcast("actor", MsgTag.EXIT, SessionType.NOTIFICATION)
        self._logger.info("Exiting...")
