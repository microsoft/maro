# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from maro.rl.model import PolicyNetForDiscreteActionSpace
from maro.rl.policy import AbsCorePolicy
from maro.rl.utils import get_truncated_cumulative_reward


class PolicyGradientConfig:
    """Configuration for the Policy Gradient algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
    """
    __slots__ = ["reward_discount"]

    def __init__(self, reward_discount: float):
        self.reward_discount = reward_discount


class PolicyGradient(AbsCorePolicy):
    """The vanilla Policy Gradient (VPG) algorithm, a.k.a., REINFORCE.

    Reference: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.

    Args:
        model (ParameterizedPolicy): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        config (PolicyGradientConfig): Configuration for the PG algorithm.
        experience_memory_size (int): Size of the experience memory. If it is -1, the experience memory is of
            unlimited size.
        experience_memory_overwrite_type (str): A string indicating how experiences in the experience memory are
            to be overwritten after its capacity has been reached. Must be "rolling" or "random".
        empty_experience_memory_after_step (bool): If True, the experience memory will be emptied  after each call
            to ``step``. Defaults to True.
        new_experience_trigger (int): Minimum number of new experiences required to trigger learning.
            Defaults to 1.
        min_experiences_to_trigger_training (int): Minimum number of experiences in the experience memory required for
            training. Defaults to 1.
    """
    def __init__(
        self,
        model: PolicyNetForDiscreteActionSpace,
        config: PolicyGradientConfig,
        experience_memory_size: int,
        experience_memory_overwrite_type: str,
        empty_experience_memory_after_step: bool = True,
        new_experience_trigger: int = 1,
        min_experiences_to_trigger_training: int = 1
    ):  
        if not isinstance(model, PolicyNetForDiscreteActionSpace):
            raise TypeError("model must be an instance of 'PolicyNetForDiscreteActionSpace'")
        super().__init__(
            model, config, experience_memory_size, experience_memory_overwrite_type,
            empty_experience_memory_after_step,
            new_experience_trigger=new_experience_trigger,
            min_experiences_to_trigger_training=min_experiences_to_trigger_training
        )

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
        if not isinstance(experience_set, ExperienceSet):
            raise TypeError(f"Expected experience object of type ExperienceSet, got {type(experience_set)}")

        states = experience_set.states
        actions = torch.from_numpy(np.asarray([act[0] for act in experience_set.actions]))
        log_p = torch.from_numpy(np.asarray([act[1] for act in experience_set.actions]))
        rewards = torch.from_numpy(np.asarray(experience_set.rewards))
        returns = get_truncated_cumulative_reward(rewards, self.special_config.reward_discount)
        returns = torch.from_numpy(returns).to(self.device)
        loss = -(log_p * returns).mean()
        self.model.step(loss)
