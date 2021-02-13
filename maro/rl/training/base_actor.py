# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

from maro.communication import Message, Proxy
from maro.utils import InternalLogger

from .abs_rollout_executor import AbsRolloutExecutor
from .message_enums import MessageTag, PayloadKey


class BaseActor(object):
    """Actor class.

    Args:
        group_name (str): Identifier of the group to which the actor belongs. It must be the same group name
            assigned to the learner (and roll-out clients, if any).
        rollout_executor (AbsRolloutExecutor): Roll-out executor.
        proxy_options (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
    """
    def __init__(self, group_name: str, rollout_executor: AbsRolloutExecutor, proxy_options: dict = None):
        self.rollout_executor = rollout_executor
        if proxy_options is None:
            proxy_options = {}
        self._proxy = Proxy(group_name, "actor", {"learner": 1}, **proxy_options)
        self._logger = InternalLogger(self._proxy.component_name)

    def run(self):
        """Entry point method.

        An infinite loop of receiving roll-out requests and performing roll-outs until and EXIT command is received.
        """
        for msg in self._proxy.receive():
            if msg.tag == MessageTag.EXIT:
                self.exit()
            elif msg.tag == MessageTag.ROLLOUT:
                ep = msg.payload[PayloadKey.ROLLOUT_INDEX]
                self._logger.info(f"Rolling out ({ep})...")
                rollout_data = self.rollout_executor.roll_out(
                    ep, training=msg.payload[PayloadKey.TRAINING], **msg.payload[PayloadKey.ROLLOUT_KWARGS]
                )
                if rollout_data is None:
                    self._logger.info(f"Roll-out {ep} aborted")
                else:
                    self._logger.info(f"Roll-out {ep} finished")
                    rollout_finish_msg = Message(
                        MessageTag.FINISHED,
                        self._proxy.component_name,
                        self._proxy.peers_name["learner"][0],
                        payload={
                            PayloadKey.ROLLOUT_INDEX: ep,
                            PayloadKey.METRICS: self.rollout_executor.env.metrics,
                            PayloadKey.DETAILS: rollout_data
                        }
                    )
                    self._proxy.isend(rollout_finish_msg)

    def exit(self):
        self._logger.info("Exiting...")
        sys.exit(0)
