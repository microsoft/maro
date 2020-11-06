# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from collections import defaultdict
from enum import Enum

from maro.communication import Proxy, SessionType
from maro.communication.registry_table import RegisterTable
from maro.rl.actor.abs_actor import AbsActor
from maro.rl.dist_topologies.common import PayloadKey


class MessageTag(Enum):
    ROLLOUT = "rollout"
    UPDATE = "update"


class ActorProxy(object):
    """A simple proxy wrapper for sending roll-out requests to remote actors.

    Args:
        proxy_params: Parameters for instantiating a ``Proxy`` instance.
    """
    def __init__(self, proxy_params):
        self._proxy = Proxy(component_type="learner", **proxy_params)

    def roll_out(
        self, model_dict: dict = None, epsilon_dict: dict = None, done: bool = False, return_details: bool = True
    ):
        """Send roll-out requests to remote actors.

        This method has exactly the same signature as ``SimpleActor``'s ``roll_out`` method but instead of doing
        the roll-out itself, sends roll-out requests to remote actors and returns the results sent back. The
        ``SimpleLearner`` simply calls the actor's ``roll_out`` method without knowing whether its performed locally
        or remotely.

        Args:
            model_dict (dict): If not None, the agents will load the models from model_dict and use these models
                to perform roll-out.
            epsilon_dict (dict): Exploration rate by agent.
            done (bool): If True, the current call is the last call, i.e., no more roll-outs will be performed.
                This flag is used to signal remote actor workers to exit.
            return_details (bool): If True, return experiences as well as performance metrics provided by the env.

        Returns:
            Performance and per-agent experiences from the remote actor.
        """
        if done:
            self._proxy.ibroadcast(
                tag=MessageTag.ROLLOUT,
                session_type=SessionType.NOTIFICATION,
                payload={PayloadKey.DONE: True}
            )
            return None, None
        else:
            performance, exp_by_agent = {}, {}
            payloads = [(peer, {PayloadKey.MODEL: model_dict,
                                PayloadKey.EPSILON: epsilon_dict,
                                PayloadKey.RETURN_DETAILS: return_details})
                        for peer in self._proxy.peers["actor"]]
            # TODO: double check when ack enable
            replies = self._proxy.scatter(
                tag=MessageTag.ROLLOUT,
                session_type=SessionType.TASK,
                destination_payload_list=payloads
            )
            for msg in replies:
                performance[msg.source] = msg.payload[PayloadKey.PERFORMANCE]
                if msg.payload[PayloadKey.EXPERIENCE] is not None:
                    for agent_id, exp_set in msg.payload[PayloadKey.EXPERIENCE].items():
                        if agent_id not in exp_by_agent:
                            exp_by_agent[agent_id] = defaultdict(list)
                        for k, v in exp_set.items():
                            exp_by_agent[agent_id][k].extend(v)

            return performance, exp_by_agent


class ActorWorker(object):
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

        performance, experiences = self._local_actor.roll_out(
            model_dict=data[PayloadKey.MODEL],
            epsilon_dict=data[PayloadKey.EPSILON],
            return_details=data[PayloadKey.RETURN_DETAILS]
        )

        self._proxy.reply(
            received_message=message,
            tag=MessageTag.UPDATE,
            payload={PayloadKey.PERFORMANCE: performance, PayloadKey.EXPERIENCE: experiences}
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
