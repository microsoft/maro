# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.experience.experience_memory import ExperienceMemory
from typing import Tuple

import numpy as np
import torch

from maro.rl.model import DiscretePolicyNet
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
        policy_net (DiscretePolicyNet): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        experience_memory (ExperienceMemory): An experience manager for storing and retrieving experiences
            for training.
        config (PolicyGradientConfig): Configuration for the PG algorithm.
    """
    def __init__(
        self, policy_net: DiscretePolicyNet, experience_memory: ExperienceMemory, config: PolicyGradientConfig,
    ):  
        if not isinstance(policy_net, DiscretePolicyNet):
            raise TypeError("model must be an instance of 'DiscretePolicyNet'")
        super().__init__(experience_memory)
        self.policy_net = policy_net
        self.config = config
        self.device = self.policy_net.device

    def choose_action(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return actions and log probabilities for given states."""
        with torch.no_grad():
            actions, log_p = self.policy_net.get_action(states)
        actions, log_p = actions.cpu().numpy(), log_p.cpu().numpy()
        return (actions[0], log_p[0]) if len(actions) == 1 else actions, log_p

    def update(self):
        """
        This should be called at the end of a simulation episode and the experiences obtained from
        the experience manager's ``get`` method should be a sequential set, i.e., in the order in
        which they are generated during the simulation. Otherwise, the return values may be meaningless. 
        """
        self.policy_net.train()
        experience_set = self.experience_memory.get()
        log_p = torch.from_numpy(np.asarray([act[1] for act in experience_set.actions])).to(self.device)
        rewards = torch.from_numpy(np.asarray(experience_set.rewards)).to(self.device)
        returns = get_truncated_cumulative_reward(rewards, self.config.reward_discount)
        returns = torch.from_numpy(returns).to(self.device)
        loss = -(log_p * returns).mean()
        self.policy_net.step(loss)

    def set_state(self, policy_state):
        self.policy_net.load_state_dict(policy_state)

    def get_state(self):
        return self.policy_net.state_dict()
