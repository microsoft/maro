# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from enum import Enum
from typing import Callable

from maro.communication import Proxy, SessionType
from maro.communication.registry_table import RegisterTable
from maro.rl.actor.abs_actor import AbsActor
from maro.rl.dist_topologies.common import PayloadKey


class MessageTag(Enum):
    ROLLOUT = "rollout"
    UPDATE = "update"


class LearnerWorker(object):
    """A ``AbsActor`` wrapper that accepts roll-out requests and performs roll-out tasks.

    Args:
        local_actor: An ``AbsActor`` instance.
        proxy_params: Parameters for instantiating a ``Proxy`` instance.
    """
    def __init__(self, local_actor: AbsActor, proxy_params):
        self._local_actor = local_actor
        self._proxy = Proxy(component_type="actor", **proxy_params)
        self._registry_table = RegisterTable(self._proxy.get_peers)
        self._registry_table.register_event_handler("learner:rollout:1", self.on_rollout_request)

    def on_rollout_request(self, message):
        """Perform local roll-out and send the results back to the request sender.

        Args:
            message: Message containing roll-out parameters and options.
        """
        data = message.payload
        if data.get(PayloadKey.DONE, False):
            sys.exit(0)

        performance, details = self._local_actor.roll_out(
            model_dict=data[PayloadKey.MODEL],
            epsilon_dict=data[PayloadKey.EPSILON],
            return_details=data[PayloadKey.RETURN_DETAILS]
        )

        self._proxy.reply(
            received_message=message,
            tag=MessageTag.UPDATE,
            payload={
                PayloadKey.PERFORMANCE: performance,
                PayloadKey.DETAILS: details
            }
        )

    def launch(self):
        """Entry point method.

        This enters the actor into an infinite loop of listening to requests and handling them according to the
        register table. In this case, the only type of requests the actor needs to handle is roll-out requests.
        """
        for msg in self._proxy.receive():
            self._registry_table.push(msg)
            triggered_events = self._registry_table.get()
            for handler_fn, cached_messages in triggered_events:
                handler_fn(cached_messages)
