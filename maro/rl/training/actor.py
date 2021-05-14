# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getcwd
from typing import Union

from maro.communication import Message, Proxy
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.simulator import Env
from maro.utils import Logger

from .message_enums import MessageTag, PayloadKey


class Actor(object):
    """Actor class that performs roll-out tasks.

    Args:
        env (Env): An environment instance.
        agent (Union[AbsAgent, MultiAgentWrapper]): Agent that interacts with the environment.
        mode (str): One of "local" and "distributed". Defaults to "local".
    """
    def __init__(
        self,
        env: Env,
        agent: Union[AbsAgent, MultiAgentWrapper],
        trajectory_cls,
        trajectory_kwargs: dict = None
    ):
        super().__init__()
        self.env = env
        self.agent = agent
        if trajectory_kwargs is None:
            trajectory_kwargs = {}
        self.trajectory = trajectory_cls(self.env, **trajectory_kwargs)

    def roll_out(self, index: int, training: bool = True, model_by_agent: dict = None, exploration_params=None):
        """Perform one episode of roll-out.
        Args:
            index (int): Externally designated index to identify the roll-out round.
            training (bool): If true, the roll-out is for training purposes, which usually means
                some kind of training data, e.g., experiences, needs to be collected. Defaults to True.
            model_by_agent (dict): Models to use for inference. Defaults to None.
            exploration_params: Exploration parameters to use for the current roll-out. Defaults to None.
        Returns:
            Data collected during the episode.
        """
        self.env.reset()
        self.trajectory.reset()
        if model_by_agent:
            self.agent.load_model(model_by_agent)
        if exploration_params:
            self.agent.set_exploration_params(exploration_params)

        _, event, is_done = self.env.step(None)
        while not is_done:
            state_by_agent = self.trajectory.get_state(event)
            action_by_agent = self.agent.choose_action(state_by_agent)
            env_action = self.trajectory.get_action(action_by_agent, event)
            if len(env_action) == 1:
                env_action = list(env_action.values())[0]
            _, next_event, is_done = self.env.step(env_action)
            reward = self.trajectory.get_reward()
            self.trajectory.on_env_feedback(
                event, state_by_agent, action_by_agent, reward if reward is not None else self.env.metrics
            )
            event = next_event

        return self.env.metrics, self.trajectory.on_finish() if training else None

    def as_worker(self, group: str, proxy_options=None, log_dir: str = getcwd()):
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
        logger = Logger(proxy.name, dump_folder=log_dir)
        for msg in proxy.receive():
            if msg.tag == MessageTag.EXIT:
                logger.info("Exiting...")
                proxy.close()
                sys.exit(0)
            elif msg.tag == MessageTag.ROLLOUT:
                ep = msg.payload[PayloadKey.ROLLOUT_INDEX]
                logger.info(f"Rolling out ({ep})...")
                metrics, rollout_data = self.roll_out(
                    ep,
                    training=msg.payload[PayloadKey.TRAINING],
                    model_by_agent=msg.payload[PayloadKey.MODEL],
                    exploration_params=msg.payload[PayloadKey.EXPLORATION_PARAMS]
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
                            PayloadKey.METRICS: metrics,
                            PayloadKey.DETAILS: rollout_data
                        }
                    )
                    proxy.isend(rollout_finish_msg)
                self.env.reset()
