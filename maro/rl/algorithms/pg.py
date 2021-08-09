# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple
from typing import Tuple

import numpy as np
import torch

from maro.rl.algorithms import AbsAlgorithm
from maro.rl.experience import ExperienceBatch
from maro.rl.model import DiscretePolicyNet
from maro.rl.utils import get_truncated_cumulative_reward

PGLossInfo = namedtuple("PGLossInfo", ["loss", "grad"])


class PolicyGradient(AbsAlgorithm):
    """The vanilla Policy Gradient (VPG) algorithm, a.k.a., REINFORCE.

    Reference: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.

    Args:
        policy_net (DiscretePolicyNet): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        reward_discount (float): Reward decay as defined in standard RL terminology.
    """
    def __init__(self, policy_net: DiscretePolicyNet, reward_discount: float):
        if not isinstance(policy_net, DiscretePolicyNet):
            raise TypeError("model must be an instance of 'DiscretePolicyNet'")
        super().__init__()
        self.policy_net = policy_net
        self.reward_discount = reward_discount
        self._num_learn_calls = 0
        self.device = self.policy_net.device

    def choose_action(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return actions and log probabilities for given states."""
        self.policy_net.eval()
        with torch.no_grad():
            actions, log_p = self.policy_net.get_action(states)
        actions, log_p = actions.cpu().numpy(), log_p.cpu().numpy()
        return (actions[0], log_p[0]) if len(actions) == 1 else actions, log_p

    def apply(self, grad_dict: dict):
        """Apply gradients to the underlying parameterized model."""
        self.policy_net.apply(grad_dict)

    def learn(self, batch: ExperienceBatch, inplace: bool = True):
        """
        This should be called at the end of a simulation episode and the experiences obtained from
        the experience store's ``get`` method should be a sequential set, i.e., in the order in
        which they are generated during the simulation. Otherwise, the return values may be meaningless.
        """
        assert self.policy_net.trainable, "policy_net needs to have at least one optimizer registered."
        self.policy_net.train()
        log_p = torch.from_numpy(np.asarray([act[1] for act in batch.data.actions])).to(self.device)
        rewards = torch.from_numpy(np.asarray(batch.data.rewards)).to(self.device)
        returns = get_truncated_cumulative_reward(rewards, self.reward_discount)
        returns = torch.from_numpy(returns).to(self.device)
        loss = -(log_p * returns).mean()

        if inplace:
            self.policy_net.step(loss)
            grad = None
        else:
            grad = self.policy_net.get_gradients(loss)

        return PGLossInfo(loss, grad), batch.indexes

    def set_state(self, policy_state):
        self.policy_net.load_state_dict(policy_state)

    def get_state(self):
        return self.policy_net.state_dict()
