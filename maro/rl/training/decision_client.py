# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.communication import Message, Proxy

from .message_enums import MessageTag, PayloadKey


class DecisionClient(object):
    """Get decisions from a remote learner.

    Args:
        group_name (str): Identifier of the group to which it belongs. It must be the same group name
            assigned to the learner and actors.
        receive_action_timeout (int): Timeout for receiving an action from the inference learner. Defaults to None.
        max_receive_action_attempts (int): Maximum number of attempts to receive an action. Defaults to None.
        proxy_options (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
    """
    def __init__(
        self,
        group_name: str,
        receive_action_timeout: int = None,
        max_receive_action_attempts: int = None,
        proxy_options: dict = None
    ):
        if proxy_options is None:
            proxy_options = {}
        self._proxy = Proxy(group_name, "decision_client", {"learner": 1}, **proxy_options)
        self._receive_action_timeout = receive_action_timeout
        self._max_receive_action_attempts = max_receive_action_attempts
        self._rollout_index = None  # used for request-response matching purposes.
        self._time_step = 0  # used for request-response matching purposes.

    def __getitem__(self, agent_id: str):
        return self

    @property
    def rollout_index(self):
        return self._rollout_index

    @rollout_index.setter
    def rollout_index(self, index):
        self._rollout_index = index
        self._time_step = 0

    def choose_action(self, state_by_agent: dict):
        """Get an action decision from a remote learner.

        Args:
            state_by_agent: Environment state based on which the action decision is to be made.
        """
        payload = {
            PayloadKey.STATE: state_by_agent,
            PayloadKey.ROLLOUT_INDEX: self._rollout_index,
            PayloadKey.TIME_STEP: self._time_step,
        }
        self._proxy.isend(
            Message(
                tag=MessageTag.CHOOSE_ACTION,
                source=self._proxy.name,
                destination=self._proxy.peers_name["learner"][0],
                payload=payload
            )
        )
        attempts = self._max_receive_action_attempts
        for msg in self._proxy.receive(timeout=self._receive_action_timeout):
            if msg:
                ep, t = msg.payload[PayloadKey.ROLLOUT_INDEX], msg.payload[PayloadKey.TIME_STEP]
                if msg.tag == MessageTag.ACTION and ep == rollout_index and t == time_step:
                    return msg.payload[PayloadKey.ACTION]
            elif attempts:
                # Did not receive expected reply before timeout
                attempts -= 1
                if attempts == 0:
                    return

        self._time_step += 1
