# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

from maro.communication.registry_table import RegisterTable
from maro.rl.actor.abs_actor import AbsActor

from .common import MessageTag, PayloadKey


class Worker(object):
    """Accept task requests and performs tasks.

    Args:
        executor: Object that carries out tasks.
        proxy: A Proxy instance responsible for receiving and sending messages.
    """
    def __init__(self, executor, proxy):
        self._executor = executor
        self._proxy = proxy
        self._registry_table = RegisterTable(self._proxy.peers_name)

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


class RolloutWorker(Worker):
    def __init__(self, actor: AbsActor, proxy):
        super().__init__(actor, proxy)
        self._registry_table.register_event_handler("learner:rollout:1", self.on_rollout_request)

    def on_rollout_request(self, message):
        """Perform local roll-out and send the results back to the request sender.

        Args:
            message: Message containing roll-out parameters and options.
        """
        data = message.payload
        if data.get(PayloadKey.DONE, False):
            sys.exit(0)

        performance, details = self._executor.roll_out(
            model_dict=data[PayloadKey.MODEL],
            exploration_params=data[PayloadKey.EXPLORATION_PARAMS],
            return_details=data[PayloadKey.RETURN_DETAILS]
        )

        self._proxy.reply(
            received_message=message,
            tag=MessageTag.UPDATE,
            payload={PayloadKey.PERFORMANCE: performance, PayloadKey.DETAILS: details}
        )


class ActionWorker(Worker):
    def __init__(self, agents: dict, proxy):
        super().__init__(agents, proxy)
        self._registry_table.register_event_handler("action:choose_action:1", self.get_action)

    def get_action(self, message):
        state, agent_id = message.payload[PayloadKey.STATE], message.payload[PayloadKey.AGENT_ID]
        model_action = self._executor[agent_id].choose_action(state)
        self._proxy.reply(
            received_message=message,
            tag=MessageTag.ACTION,
            payload={PayloadKey.ACTION: model_action}
        )
