# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Callable, List, Union

from maro.communication import Message, Proxy, RegisterTable, SessionType
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.storage import OverwriteType, SimpleStore
from maro.utils import InternalLogger

from .message_enums import MsgTag, MsgKey


class ActorProxy(object):
    """Actor proxy that manages a set of remote actors.

    Args:
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
        update_trigger: str = None,
        replay_memory_size: int = -1,
        replay_memory_overwrite_type: OverwriteType = None
    ):
        peers = {"actor": num_actors}
        if proxy_options is None:
            proxy_options = {}
        self._proxy = Proxy(group_name, "actor_proxy", peers, **proxy_options)
        self._actors = self._proxy.peers["actor"]  # remote actor ID's
        self._registry_table = RegisterTable(self._proxy.peers)
        if update_trigger is None:
            update_trigger = len(self._actors)
        self._registry_table.register_event_handler(
            f"actor:{MsgTag.ROLLOUT_DONE.value}:{update_trigger}", self._on_rollout_finish
        )
        self.replay_memory = defaultdict(
            lambda: SimpleStore(capacity=replay_memory_size, overwrite_type=replay_memory_overwrite_type)
        )
        self.logger = InternalLogger(self._proxy.name)

    def roll_out(self, index: int, training: bool = True, model_by_agent: dict = None, exploration_params=None):
        """Collect roll-out data from remote actors.

        Args:
            index (int): Index of roll-out requests.
            training (bool): If true, the roll-out request is for training purposes.
            model_by_agent (dict): Models to be broadcast to remote actors for inference. Defaults to None.
            exploration_params: Exploration parameters to be used by the remote roll-out actors. Defaults to None.
        """
        body = {
            MsgKey.ROLLOUT_INDEX: index,
            MsgKey.TRAINING: training,
            MsgKey.MODEL: model_by_agent,
            MsgKey.EXPLORATION_PARAMS: exploration_params
        }
        self._proxy.iscatter(MsgTag.ROLLOUT, SessionType.TASK, [(actor, body) for actor in self._actors])
        self.logger.info(f"Sent roll-out requests to {self._actors} for ep-{index}")

        # Receive roll-out results from remote actors
        for msg in self._proxy.receive():
            if msg.body[MsgKey.ROLLOUT_INDEX] != index:
                self.logger.info(
                    f"Ignore a message of type {msg.tag} with ep {msg.body[MsgKey.ROLLOUT_INDEX]} "
                    f"(expected {index} or greater)"
                )
                continue
            if msg.tag == MsgTag.ROLLOUT_DONE:
                # If enough update messages have been received, call update() and break out of the loop to start
                # the next episode.
                result = self._registry_table.push(msg)
                if result:
                    env_metrics = result
                    break
            elif msg.tag == MsgTag.REPLAY_SYNC:
                # print(f"received exp from actor {msg.source} ")
                # print({agent_id: {k: len(v) for k, v in exp.items()} for agent_id, exp in msg.body[MsgKey.REPLAY].items()})
                for agent_id, exp in msg.body[MsgKey.REPLAY].items():
                    self.replay_memory[agent_id].put(exp)
                # print({agent_id: len(pool) for agent_id, pool in self.replay_memory.items()})

        return env_metrics

    def _on_rollout_finish(self, messages: List[Message]):
        metrics = {msg.source: msg.body[MsgKey.METRICS] for msg in messages}
        for msg in messages:
            for agent_id, replay in msg.body[MsgKey.REPLAY].items():
                self.replay_memory[agent_id].put(replay)
        return metrics

    def terminate(self):
        """Tell the remote actors to exit."""
        self._proxy.ibroadcast(
            component_type="actor", tag=MsgTag.EXIT, session_type=SessionType.NOTIFICATION
        )
        self.logger.info("Exiting...")
