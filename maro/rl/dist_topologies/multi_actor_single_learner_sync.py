# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from collections import defaultdict
import sys

from maro.communication import Proxy, SessionType
from maro.communication.registry_table import RegisterTable
from maro.rl.actor.abstract_actor import RolloutMode
from maro.rl.dist_topologies.common import PayloadKey


class MessageType(Enum):
    ROLLOUT = "rollout"
    UPDATE = "update"


class ActorProxy(object):
    def __init__(self, proxy_params):
        self._proxy = Proxy(component_type="actor_proxy", **proxy_params)

    def roll_out(self, mode: RolloutMode, models: dict = None, epsilon_dict: dict = None, seed: int = None):
        if mode == RolloutMode.EXIT:
            # TODO: session type: notification
            self._proxy.broadcast(tag=MessageType.ROLLOUT,
                                  session_type=SessionType.TASK,
                                  payload={PayloadKey.RolloutMode: mode})
            return None, None
        else:
            performance, exp_by_agent = {}, {}
            payloads = [(peer, {PayloadKey.MODEL: models,
                                PayloadKey.RolloutMode: mode,
                                PayloadKey.EPSILON: epsilon_dict,
                                PayloadKey.SEED: (seed+i) if seed is not None else None})
                        for i, peer in enumerate(self._proxy.get_peers("actor_worker"))]
            # TODO: double check when ack enable
            replies = self._proxy.scatter(tag=MessageType.ROLLOUT, session_type=SessionType.TASK,
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
    def __init__(self, local_actor, proxy_params):
        self._local_actor = local_actor
        self._proxy = Proxy(component_type="actor_worker", **proxy_params)
        self._registry_table = RegisterTable(self._proxy.get_peers)
        self._registry_table.register_event_handler("actor_proxy:rollout:1", self.on_rollout_request)

    def on_rollout_request(self, message):
        if message.payload[PayloadKey.RolloutMode] == RolloutMode.EXIT:
            sys.exit(0)

        data = message.payload
        performance, exp_by_agent = self._local_actor.roll_out(mode=data[PayloadKey.RolloutMode],
                                                               models=data[PayloadKey.MODEL],
                                                               epsilon_dict=data[PayloadKey.EPSILON],
                                                               seed=data[PayloadKey.SEED])

        self._proxy.reply(received_message=message,
                          tag=MessageType.UPDATE,
                          payload={PayloadKey.PERFORMANCE: performance["local"],
                                   PayloadKey.EXPERIENCE: exp_by_agent}
                          )

    def launch(self):
        # TODO
        """
        Launches the an ActorWorker instance
        """
        for msg in self._proxy.receive():
            self._registry_table.push(msg)
            triggered_events = self._registry_table.get()
            for handler_fn, cached_messages in triggered_events:
                handler_fn(cached_messages)
