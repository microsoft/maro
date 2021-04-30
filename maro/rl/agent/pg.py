# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from maro.rl.model import SimpleMultiHeadModel
from maro.rl.storage import SimpleStore
from maro.rl.utils import get_truncated_cumulative_reward

from .agent import AbsAgent, GenericAgentConfig


class PolicyGradientConfig:
    """Configuration for the Policy Gradient algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
    """
    __slots__ = ["reward_discount"]

    def __init__(self, reward_discount: float):
        self.reward_discount = reward_discount


class PolicyGradient(AbsAgent):
    """The vanilla Policy Gradient (VPG) algorithm, a.k.a., REINFORCE.

    Reference: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.

    Args:
        model (AbsCoreModel): Task model or container of task models required by the algorithm.
        algorithm_config: Algorithm-specific configuration.
        generic_config (GenericAgentConfig): Non-algorithm-specific configuration.  
        experience_memory (SimpleStore): Experience memory for the agent. If None, an experience memory will be
            created at init time. Defaults to None.
    """
    def choose_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use the actor (policy) model to generate stochastic actions.

        Args:
            state: Input to the actor model.

        Returns:
            Actions and corresponding log probabilities.
        """
        state = torch.from_numpy(state).to(self.device)
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(dim=0)

        action_prob = Categorical(self.model(state, training=False))
        action = action_prob.sample()
        log_p = action_prob.log_prob(action)
        action, log_p = action.cpu().numpy(), log_p.cpu().numpy()
        return (action[0], log_p[0]) if is_single else (action, log_p)

    def step(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        returns = get_truncated_cumulative_reward(rewards, self.config.reward_discount)
        returns = torch.from_numpy(returns).to(self.device)
        action_distributions = self.model(states)
        action_prob = action_distributions.gather(1, actions.unsqueeze(1)).squeeze()   # (N, 1)
        loss = -(torch.log(action_prob) * returns).mean()
        self.model.step(loss)
