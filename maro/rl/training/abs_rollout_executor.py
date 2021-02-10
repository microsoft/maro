# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Union

from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.shaping import Shaper
from maro.simulator import Env

from .decision_client import DecisionClient


class AbsRolloutExecutor(ABC):
    """Abstract rollout executor class.
    Args:
        env (Env): An environment instance.
        agent (Union[AbsAgent, MultiAgentWrapper, DecisionClient]): Agent that interacts with the environment.
            If it is a ``DecisionClient``, action decisions will be obtained from a remote inference learner.
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
        agent: Union[AbsAgent, MultiAgentWrapper, DecisionClient],
        state_shaper: Shaper = None,
        action_shaper: Shaper = None,
        experience_shaper: Shaper = None
    ):
        self.env = env
        self.agent = agent
        self.state_shaper = state_shaper
        self.action_shaper = action_shaper
        self.experience_shaper = experience_shaper

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
