# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Union

from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.simulator import Env

from .decision_client import DecisionClient


class AbsRolloutExecutor(ABC):
    """Abstract rollout executor class.
    Args:
        env (Env): An environment instance.
        agent (Union[AbsAgent, MultiAgentWrapper, DecisionClient]): Agent that interacts with the environment.
            If it is a ``DecisionClient``, action decisions will be obtained from a remote inference learner.
    """
    def __init__(self, env: Env, agent: Union[AbsAgent, MultiAgentWrapper, DecisionClient]):
        self.env = env
        self.agent = agent

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
