# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from maro.communication import Proxy, SessionMessage, SessionType

from .common import MessageTag, PayloadKey


class RolloutProxy(Proxy):
    """A simple proxy wrapper for sending roll-out requests to remote actors.

    Args:
        proxy_params: Parameters for instantiating a ``Proxy`` instance.
        experience_collecting_func (Callable): A function responsible for collecting experiences from multiple sources.
    """
    def __init__(self, proxy_params, experience_collecting_func: Callable):
        super().__init__(component_type="roll_out", **proxy_params)
        self._experience_collecting_func = experience_collecting_func

    def roll_out(
        self, model_dict: dict = None, exploration_params=None, done: bool = False, return_details: bool = True
    ):
        """Send roll-out requests to remote actors.

        This method has exactly the same signature as ``SimpleActor``'s ``roll_out`` method but instead of doing
        the roll-out itself, sends roll-out requests to remote actors and returns the results sent back. The
        ``SimpleLearner`` simply calls the actor's ``roll_out`` method without knowing whether its performed locally
        or remotely.

        Args:
            model_dict (dict): If not None, the agents will load the models from model_dict and use these models
                to perform roll-out.
            exploration_params: Exploration parameters.
            done (bool): If True, the current call is the last call, i.e., no more roll-outs will be performed.
                This flag is used to signal remote actor workers to exit.
            return_details (bool): If True, return experiences as well as performance metrics provided by the env.

        Returns:
            Performance and per-agent experiences from the remote actor.
        """
        if done:
            self.ibroadcast(
                component_type="actor",
                tag=MessageTag.ROLLOUT,
                session_type=SessionType.NOTIFICATION,
                payload={PayloadKey.DONE: True}
            )
            return None, None

        payloads = [(peer, {PayloadKey.MODEL: model_dict,
                            PayloadKey.EXPLORATION_PARAMS: exploration_params,
                            PayloadKey.RETURN_DETAILS: return_details})
                    for peer in self.peers_name["actor"]]
        # TODO: double check when ack enable
        replies = self.scatter(
            tag=MessageTag.ROLLOUT,
            session_type=SessionType.TASK,
            destination_payload_list=payloads
        )

        performance = [(msg.source, msg.payload[PayloadKey.PERFORMANCE]) for msg in replies]
        details_by_source = {msg.source: msg.payload[PayloadKey.DETAILS] for msg in replies}
        details = self._experience_collecting_func(details_by_source) if return_details else None

        return performance, details


class ActionProxy(Proxy):
    def __init__(self, proxy_params):
        super().__init__(component_type="action", **proxy_params)

    def choose_action(self, state, agent_id):
        reply = self.send(
            SessionMessage(
                tag=MessageTag.CHOOSE_ACTION,
                source=self.component_name,
                destination=self.peers_name["action_server"],
                payload={PayloadKey.STATE: state, PayloadKey.AGENT_ID: agent_id},
            )
        )

        return reply.payload[PayloadKey.ACTION]
