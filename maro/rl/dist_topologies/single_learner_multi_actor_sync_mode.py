# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from collections import defaultdict
import sys

from maro.communication import Proxy, SessionType
from maro.communication.registry_table import RegisterTable
from maro.rl.dist_topologies.common import PayloadKey
from maro.rl.actor.abs_actor import AbsActor


class MessageTag(Enum):
    ROLLOUT = "rollout"
    UPDATE = "update"


class ActorProxy(object):
    def __init__(self, proxy_params):
        self._proxy = Proxy(component_type="actor", **proxy_params)

    def roll_out(self, model_dict: dict = None, epsilon_dict: dict = None, done: bool = False,
                 return_details: bool = True):
        if done:
            self._proxy.ibroadcast(tag=MessageTag.ROLLOUT,
                                   session_type=SessionType.NOTIFICATION,
                                   payload={PayloadKey.DONE: True})
            return None, None
        else:
            performance, exp_by_agent = {}, {}
            payloads = [(peer, {PayloadKey.MODEL: model_dict,
                                PayloadKey.EPSILON: epsilon_dict,
                                PayloadKey.RETURN_DETAILS: return_details})
                        for peer in self._proxy.peers["actor_worker"]]
            # TODO: double check when ack enable
            replies = self._proxy.scatter(tag=MessageTag.ROLLOUT, session_type=SessionType.TASK,
                                          destination_payload_list=payloads)
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
    def __init__(self, local_actor: AbsActor, proxy_params):
        self._local_actor = local_actor
        self._proxy = Proxy(component_type="actor_worker", **proxy_params)
        self._registry_table = RegisterTable(self._proxy.get_peers)
        self._registry_table.register_event_handler("actor:rollout:1", self.on_rollout_request)

    def on_rollout_request(self, message):
        data = message.payload
        if data.get(PayloadKey.DONE, False):
            sys.exit(0)

        performance, experiences = self._local_actor.roll_out(model_dict=data[PayloadKey.MODEL],
                                                              epsilon_dict=data[PayloadKey.EPSILON],
                                                              return_details=data[PayloadKey.RETURN_DETAILS])

        self._proxy.reply(received_message=message,
                          tag=MessageTag.UPDATE,
                          payload={PayloadKey.PERFORMANCE: performance,
                                   PayloadKey.EXPERIENCE: experiences}
                          )

    def launch(self):
        """
        This launches an ActorWorker instance.
        """
        for msg in self._proxy.receive():
            self._registry_table.push(msg)
            triggered_events = self._registry_table.get()
            for handler_fn, cached_messages in triggered_events:
                handler_fn(cached_messages)
