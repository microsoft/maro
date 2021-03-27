# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, List, Union

from maro.communication import Message, Proxy, RegisterTable, SessionType
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.storage import OverwriteType, SimpleStore
from maro.rl.training import AbsEnvWrapper
from maro.utils import InternalLogger

from .message_enums import MsgTag, MsgKey


class Actor(object):
    """On-demand roll-out executor.

    Args:
        env (AbsEnvWrapper): An ``AbsEnvWrapper`` instance that wraps an ``Env`` instance with scenario-specific
            processing logic and stores transitions during roll-outs in a replay memory.
        agent (Union[AbsAgent, MultiAgentWrapper]): Agent that interacts with the environment.
        group (str): Identifier of the group to which the actor belongs. It must be the same group name
            assigned to the learner (and decision clients, if any).
        proxy_options (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
        replay_sync_interval (int): Number of roll-out steps between replay syncing calls.
    """
    def __init__(
        self,
        env: AbsEnvWrapper,
        agent: Union[AbsAgent, MultiAgentWrapper],
        group: str,
        proxy_options: dict = None,
        replay_sync_interval: int = None,
        send_results: bool = True
    ):
        if replay_sync_interval == 0:
            raise ValueError("replay_sync_interval must be a positive integer or None")
        self.env = env
        self.agent = MultiAgentWrapper(agent) if isinstance(agent, AbsAgent) else agent
        if proxy_options is None:
            proxy_options = {}
        self._proxy = Proxy(group, "actor", {"actor_proxy": 1}, **proxy_options)
        self.replay_sync_interval = replay_sync_interval
        self.send_results = send_results
        self._logger = InternalLogger(self._proxy.name)

    def roll_out(self, index: int, training: bool = True, model_by_agent: dict = None, exploration_params=None):
        self.env.reset()
        if not training:
            self.env.record_path = False  # no need to record the trajectory if roll-out is not for training

        # Load models and exploration parameters
        if model_by_agent:
            self.agent.load_model(model_by_agent)
        if exploration_params:
            self.agent.set_exploration_params(exploration_params)

        state = self.env.start(rollout_index=index)  # get initial state
        while state:
            action = self.agent.choose_action(state)
            state = self.env.step(action)
            if self.replay_sync_interval is not None and (self.env.step_index + 1) % self.replay_sync_interval == 0:
                self._proxy.isend(
                    Message(
                        MsgTag.REPLAY_SYNC, self._proxy.name, self._proxy.peers["actor_proxy"][0],
                        body={MsgKey.ROLLOUT_INDEX: index, MsgKey.REPLAY: self.env.replay_memory}
                    )
                )
                self.env.flush()

        return self.env.metrics 

    def run(self):
        for msg in self._proxy.receive():
            if msg.tag == MsgTag.EXIT:
                self._logger.info("Exiting...")
                break
            elif msg.tag == MsgTag.ROLLOUT:
                self.on_rollout_request(msg)
                self.post_rollout(msg.body[MsgKey.ROLLOUT_INDEX])

    def on_rollout_request(self, msg: Message):
        ep = msg.body[MsgKey.ROLLOUT_INDEX]
        self._logger.info(f"Rolling out ({ep})...")
        self.roll_out(
            ep,
            training=msg.body[MsgKey.TRAINING],
            model_by_agent=msg.body[MsgKey.MODEL],
            exploration_params=msg.body[MsgKey.EXPLORATION_PARAMS]
        )
        self._logger.info(f"Roll-out {ep} finished")

    def post_rollout(self, index: int):
        if not self.send_results:
            return

        body = {MsgKey.ROLLOUT_INDEX: index, MsgKey.METRICS: self.env.metrics}
        if self.env.record_path:
            body[MsgKey.REPLAY] = self.env.replay_memory

        actor_proxy_addr = self._proxy.peers["actor_proxy"][0]
        self._proxy.isend(
            Message(MsgTag.ROLLOUT_DONE, self._proxy.name, actor_proxy_addr, body=body)
        )
