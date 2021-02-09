# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from maro.communication import Message, Proxy
from maro.rl.shaping import Shaper
from maro.simulator import Env

from .message_enums import MessageTag, PayloadKey


class RolloutClient(object):
    """A mirror of ``AbsRolloutExecutor`` that uses a proxy to query for actions from a remote learner.

    Args:
        group_name (str): Identifier of the group to which it belongs. It must be the same group name
            assigned to the learner and actors.
        env (Env): An environment instance.
        state_shaper (Shaper, optional): It is responsible for converting the environment observation to model
            input. Defaults to None.
        action_shaper (Shaper, optional): It is responsible for converting an agent's model output to environment
            executable action. Defaults to None.
        experience_shaper (Shaper, optional): It is responsible for processing data in the replay buffer at
            the end of an episode. Defaults to None.
        proxy_options (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
        receive_action_timeout (int): Timeout for receiving an action from the inference learner. Defaults to None.
        max_receive_action_attempts (int): Maximum number of attempts to receive an action. Defaults to None.
    """
    def __init__(
        self,
        group_name: str,
        env: Env,
        state_shaper: Shaper = None,
        action_shaper: Shaper = None,
        experience_shaper: Shaper = None,
        receive_action_timeout: int = None,
        max_receive_action_attempts: int = None,
        proxy_options: dict = None
    ):
        self.env = env
        self.state_shaper = state_shaper
        self.action_shaper = action_shaper
        self.experience_shaper = experience_shaper
        if proxy_options is None:
            proxy_options = {}
        self._proxy = Proxy(group_name, "rollout_client", {"learner": 1}, **proxy_options)
        self._receive_action_timeout = receive_action_timeout
        self._max_receive_action_attempts = max_receive_action_attempts

    @abstractmethod
    def roll_out(self, index: int, training: bool = True, **kwargs):
        """Perform one episode of roll-out.

        Args:
            index (int): Externally designated index to identify the roll-out round.
            training (bool): If true, the roll-out is for training purposes, which usually means
                some kind of training data, e.g., experiences, needs to be collected. Defaults to True.

        Returns:
            Data collected during the episode.
        """
        raise NotImplementedError

    def get_action(self, state, rollout_index: int, time_step: int, agent_id: str = None):
        """Get an action decision from the remote agent.

        Args:
            state: Environment state based on which the actin decision is to be made.
            rollout_index (int): The roll-out index to which the current action decision belongs to.
                This is used for request-response matching purposes.
            time_step (int): The time step index at which the action decision occurs. This is used for
                request-response matching purposes.
            agent_id (str): The name of the agent to make the action decision. Defaults to None.
        """
        assert rollout_index is not None and time_step is not None, "rollout_index and time_step cannot be None"
        payload = {
            PayloadKey.STATE: state,
            PayloadKey.ROLLOUT_INDEX: rollout_index,
            PayloadKey.TIME_STEP: time_step,
            PayloadKey.AGENT_ID: agent_id,
        }
        self._proxy.isend(
            Message(
                tag=MessageTag.CHOOSE_ACTION,
                source=self._proxy.component_name,
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
            else:
                # Did not receive expected reply before timeout
                attempts -= 1
            if attempts == 0:
                return
