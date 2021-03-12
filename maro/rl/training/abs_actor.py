# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from abc import ABC, abstractmethod
from typing import Union

from maro.communication import Message, Proxy
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.simulator import Env
from maro.utils import InternalLogger

from .decision_client import DecisionClient
from .message_enums import MessageTag, PayloadKey


class AbsActor(ABC):
    """Actor class that performs roll-out tasks.

    Args:
        env (Env): An environment instance.
        agent (Union[AbsAgent, MultiAgentWrapper, DecisionClient]): Agent that interacts with the environment.
            If it is a ``DecisionClient``, action decisions will be obtained from a remote inference learner.
        mode (str): One of "local" and "distributed". Defaults to "local".
        
    """
    def __init__(self, env: Env, agent: Union[AbsAgent, MultiAgentWrapper, DecisionClient]):
        super().__init__()
        self.env = env
        self.agent = agent

    @abstractmethod
    def roll_out(self, index: int, training: bool = True, **kwargs):
        """Perform one episode of roll-out.
        Args:
            index (int): Externally designated index to identify the roll-out round.
            training (bool): If true, the roll-out is for training purposes, which usually means
                some kind of training data, e.g., experiences, needs to be collected. Defaults to True.
        Returns:
            Data collected during the episode.
        """
        raise NotImplementedError

    def as_worker(self, group: str, proxy_options=None):
        """Perform roll-outs based on a remote learner's request.

        Args:
            group (str): Identifier of the group to which the actor belongs. It must be the same group name
                assigned to the learner (and decision clients, if any). If 
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
