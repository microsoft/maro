# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

from typing import Union

from maro.communication import Message, Proxy
from maro.rl.actor import AbsActor
from maro.utils import InternalLogger

from .actor_client import ActorClient
from .common import MessageTag, PayloadKey


class BaseDistActor(object):
    """Distributed actor class.

    Args:
        actor (Union[AbsActor, ActorClient]): An ``AbsActor`` instance that performs roll-outs.
        proxy (Proxy): A ``Proxy`` instance responsible for communication.
    """
    def __init__(self, actor: Union[AbsActor, ActorClient], proxy: Proxy):
        self.actor = actor
        self.proxy = proxy
        self.name = self.proxy.component_name
        self.learner_name = self.proxy.peers_name["learner"][0]
        self._logger = InternalLogger(self.proxy.component_name)

    def run(self):
        """Entry point method.

        This enters the actor into an infinite loop of receiving roll-out requests and performing roll-outs.
        """
        for msg in self.proxy.receive():
            if msg.tag == MessageTag.EXIT:
                self.exit()
            elif msg.tag == MessageTag.ROLLOUT:
                self.actor.update_agent(
                    model_dict=msg.payload.get(PayloadKey.MODEL, None),
                    exploration_params=msg.payload.get(PayloadKey.EXPLORATION_PARAMS, None)
                )
                ep = msg.payload[PayloadKey.EPISODE]
                self._logger.info(f"Rolling out for ep-{ep}...")
                performance, details = self.actor.roll_out(
                    ep, is_training=msg.payload[PayloadKey.IS_TRAINING], **msg.payload[PayloadKey.ROLLOUT_KWARGS]
                )
                self._logger.info(f"Roll-out finished for ep-{ep}")
                payload = {PayloadKey.EPISODE: ep, PayloadKey.PERFORMANCE: performance, PayloadKey.DETAILS: details}
                # If the actor is an ActorClient instance, we need to tell the learner the ID of the actor client
                # so that the learner can send termination signals to the actor clients of unfinished actors.
                if isinstance(self.actor, ActorClient):
                    payload[PayloadKey.ACTOR_CLIENT_ID] = self.actor.agent.component_name
                self.proxy.isend(Message(MessageTag.FINISHED, self.name, self.learner_name, payload=payload))

    def exit(self):
        self._logger.info("Exiting...")
        sys.exit(0)
