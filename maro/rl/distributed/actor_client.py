# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from maro.communication import Message, Proxy
from maro.rl.actor import AbsActor
from maro.rl.shaping import Shaper
from maro.simulator import Env

from .common import MessageTag, PayloadKey, TerminateRollout


class ActorClient(AbsActor):
    """Actor client class that uses a proxy to query for actions from a remote learner.

    Args:
        env (Env): An environment instance.
        agent_proxy (Proxy): A ``Proxy`` used to query the remote learner for action decisions.
        state_shaper (Shaper, optional): It is responsible for converting the environment observation to model
            input. Defaults to None.
        action_shaper (Shaper, optional): It is responsible for converting an agent's model output to environment
            executable action. Defaults to None.
        experience_shaper (Shaper, optional): It is responsible for processing data in the replay buffer at
            the end of an episode. Defaults to None.
    """
    def __init__(
        self,
        env: Env,
        agent_proxy: Proxy,
        state_shaper: Shaper = None,
        action_shaper: Shaper = None,
        experience_shaper: Shaper = None,
        receive_action_timeout: int = None,
        max_receive_action_attempts: int = None
    ):
        super().__init__(
            env, agent_proxy,
            state_shaper=state_shaper, action_shaper=action_shaper, experience_shaper=experience_shaper
        )
        self._receive_action_timeout = receive_action_timeout
        self._max_receive_action_attempts = max_receive_action_attempts

    @abstractmethod
    def roll_out(self, index: int, is_training: bool = True, **kwargs):
        """Perform one episode of roll-out.

        Args:
            index (int): Externally designated index to identify the roll-out round.
            is_training (bool): If true, the roll-out is for training purposes, which usually means
                some kind of training data, e.g., experiences, needs to be collected. Defaults to True.
        """
        raise NotImplementedError

    def get_action(self, rollout_index: int, time_step: int, state, agent_id: str = None):
        payload = {
            PayloadKey.STATE: state,
            PayloadKey.EPISODE: rollout_index,
            PayloadKey.TIME_STEP: time_step,
            PayloadKey.AGENT_ID: agent_id,
        }
        self.agent.isend(
            Message(
                tag=MessageTag.CHOOSE_ACTION,
                source=self.agent.component_name,
                destination=self.agent.peers_name["learner"][0],
                payload=payload
            )
        )
        attempts = self._max_receive_action_attempts
        for msg in self.agent.receive(timeout=self._receive_action_timeout):
            if msg:
                ep, t = msg.payload[PayloadKey.EPISODE], msg.payload[PayloadKey.TIME_STEP]
                if msg.tag == MessageTag.TERMINATE_EPISODE and ep == rollout_index:
                    return TerminateRollout()
                if msg.tag == MessageTag.ACTION and ep == rollout_index and t == time_step:
                    return msg.payload[PayloadKey.ACTION]

            # Did not receive expected reply before timeout
            attempts -= 1
            if attempts == 0:
                return
