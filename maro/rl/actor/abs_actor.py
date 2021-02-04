# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Union

from maro.communication import Proxy
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.shaping import Shaper
from maro.simulator import Env


class AbsActor(ABC):
    """Abstract actor class.

    Args:
        env (Env): An environment instance.
        agent (Union[AbsAgent, MultiAgentWrapper, Proxy]): Agent that interacts with the environment. If it is
            ``Proxy``, 
            in which case the actor will query the remote learner for action decisions.
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
        agent: Union[AbsAgent, MultiAgentWrapper],
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
    def roll_out(self, index: int, is_training: bool = True, **kwargs):
        """Perform one episode of roll-out.
        
        Args:
            index (int): Externally designated index to identify the roll-out round.
            is_training (bool): If true, the roll-out is for training purposes, which usually means
                some kind of training data, e.g., experiences, needs to be collected. Defaults to True. 
        """
        raise NotImplementedError

    def update_agent(self, model_dict: dict = None, exploration_params: dict = None):
        """Update the agent's models and exploration parameters ahead of roll-out.
        
        Args:
            model_dict (dict): Dictionary of models to be loaded for agents. Defaults to None.
            exploration_params (dict): Exploration parameters. Defaults to None.
        """
        if model_dict is not None:
            self.agent.load_model(model_dict)
        if exploration_params is not None:
            self.agent.set_exploration_params(exploration_params)
