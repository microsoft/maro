# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from typing import Callable, Union

from maro.communication import Message, Proxy
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.simulator import Env
from maro.utils import InternalLogger

from .decision_client import DecisionClient
from .message_enums import MessageTag, PayloadKey


class Actor(object):
    """Actor class that performs roll-out tasks.

    Args:
        env (Env): An environment instance.
        agent (Union[AbsAgent, MultiAgentWrapper, DecisionClient]): Agent that interacts with the environment.
            If it is a ``DecisionClient``, action decisions will be obtained from a remote inference learner.
        mode (str): One of "local" and "distributed". Defaults to "local".
    """
    def __init__(
        self,
        env: Env,
        agent: Union[AbsAgent, MultiAgentWrapper, DecisionClient],
        trajectory_cls,
        trajectory_kwargs: dict = None,
        max_null_decisions_allowed: int = None
    ):
        super().__init__()
        self.env = env
        self.agent = agent
        if trajectory_kwargs is None:
            trajectory_kwargs = {}
        self.trajectory = trajectory_cls(self.env, **trajectory_kwargs)
        if max_null_decisions_allowed is not None:
            self.max_null_decisions_allowed = max_null_decisions_allowed
        else:
            self.max_null_decisions_allowed = float("inf")

    def roll_out(self, index: int, training: bool = True, model_dict=None, exploration_params=None):
        """Perform one episode of roll-out.
        Args:
            index (int): Externally designated index to identify the roll-out round.
            training (bool): If true, the roll-out is for training purposes, which usually means
                some kind of training data, e.g., experiences, needs to be collected. Defaults to True.
        Returns:
            Data collected during the episode.
        """
        self.env.reset()
        self.trajectory.reset()
        if isinstance(self.agent, DecisionClient):
            self.agent.rollout_index = index
            null_decisions_allowed = self.max_null_decisions_allowed
        else:
            if model_dict:
                self.agent.load_model(model_dict)  
            if exploration_params:
                self.agent.set_exploration_params(exploration_params)

        _, event, is_done = self.env.step(None)
        while not is_done:
            state_by_agent = self.trajectory.get_state(event)
            action_by_agent = self.agent.choose_action(state_by_agent)
            if isinstance(self.agent, DecisionClient) and action_by_agent is None:
                self.logger.info(f"Failed to receive an action, proceed with no action.")
                if null_decisions_allowed:
                    null_decisions_allowed -= 1
                    if null_decisions_allowed == 0:
                        self.logger.info(f"Roll-out aborted after {self.max_null_decisions_allowed} null decisions.")
                        return
                env_action = None
            else:
                env_action = self.trajectory.get_action(action_by_agent, event)
            if len(env_action) == 1:
                env_action = list(env_action.values())[0]
            _, next_event, is_done = self.env.step(env_action)
            reward = self.trajectory.get_reward()
            self.trajectory.on_env_feedback(
                event, state_by_agent, action_by_agent, reward if reward is not None else self.env.metrics
            )
            event = next_event

        return self.trajectory.on_finish() if training else None

    def as_worker(self, group: str, proxy_options=None):
        """Executes an event loop where roll-outs are performed on demand from a remote learner.

        Args:
            group (str): Identifier of the group to which the actor belongs. It must be the same group name
                assigned to the learner (and decision clients, if any).
            proxy_options (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
                for details. Defaults to None.
        """
        if proxy_options is None:
            proxy_options = {}
        proxy = Proxy(group, "actor", {"learner": 1}, **proxy_options)
        logger = InternalLogger(proxy.name)
        for msg in proxy.receive():
            if msg.tag == MessageTag.EXIT:
                logger.info("Exiting...")
                sys.exit(0)
            elif msg.tag == MessageTag.ROLLOUT:
                ep = msg.payload[PayloadKey.ROLLOUT_INDEX]
                logger.info(f"Rolling out ({ep})...")
                rollout_data = self.roll_out(
                    ep, training=msg.payload[PayloadKey.TRAINING], **msg.payload[PayloadKey.ROLLOUT_KWARGS]
                )
                if rollout_data is None:
                    logger.info(f"Roll-out {ep} aborted")
                else:
                    logger.info(f"Roll-out {ep} finished")
                    rollout_finish_msg = Message(
                        MessageTag.FINISHED,
                        proxy.name,
                        proxy.peers_name["learner"][0],
                        payload={
                            PayloadKey.ROLLOUT_INDEX: ep,
                            PayloadKey.METRICS: self.env.metrics,
                            PayloadKey.DETAILS: rollout_data
                        }
                    )
                    proxy.isend(rollout_finish_msg)
                self.env.reset()
