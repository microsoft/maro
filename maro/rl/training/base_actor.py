# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

from maro.communication import Message, Proxy
from maro.utils import InternalLogger

from .abs_rollout_executor import AbsRolloutExecutor
from .message_enums import MessageTag, PayloadKey
from .rollout_client import RolloutClient


class BaseActor(object):
    """Actor class.

    Args:
        executor (AbsRolloutExecutor): An ``AbsRolloutExecutor`` instance that performs roll-outs.
        proxy (Proxy): A ``Proxy`` instance responsible for communication.
    """
    def __init__(self, executor: AbsRolloutExecutor, proxy: Proxy):
        self.executor = executor
        self.proxy = proxy
        self.name = self.proxy.component_name
        self.learner_name = self.proxy.peers_name["learner"][0]

        self._logger = InternalLogger(self.proxy.component_name)

    def run(self):
        """Entry point method.

        This enters the roll-out executor into an infinite loop of receiving roll-out requests and
        performing roll-outs.
        """
        for msg in self.proxy.receive():
            if msg.tag == MessageTag.EXIT:
                self.exit()
            elif msg.tag == MessageTag.ROLLOUT:
                self.executor.update_agent(
                    model_dict=msg.payload.get(PayloadKey.MODEL, None),
                    exploration_params=msg.payload.get(PayloadKey.EXPLORATION_PARAMS, None)
                )
                ep = msg.payload[PayloadKey.ROLLOUT_INDEX]
                self._logger.info(f"Rolling out ({ep})...")
                rollout_data = self.executor.roll_out(
                    ep, training=msg.payload[PayloadKey.TRAINING], **msg.payload[PayloadKey.ROLLOUT_KWARGS]
                )
                if rollout_data is None:
                    self._logger.info(f"Roll-out {ep} aborted")
                else:
                    self._logger.info(f"Roll-out {ep} finished")
                    payload = {
                        PayloadKey.ROLLOUT_INDEX: ep, 
                        PayloadKey.METRICS: self.executor.env.metrics,
                        PayloadKey.DETAILS: rollout_data
                    }
                    self.proxy.isend(Message(MessageTag.FINISHED, self.name, self.learner_name, payload=payload))

    def exit(self):
        self._logger.info("Exiting...")
        sys.exit(0)
