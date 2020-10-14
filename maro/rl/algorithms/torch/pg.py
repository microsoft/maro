# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import torch.nn as nn

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.utils.trajectory_utils import get_k_step_returns


class PolicyGradientHyperParameters:
    """Hyper-parameter set for the Actor-Critic algorithm.

    Args:
        num_actions (int): number of possible actions
        reward_decay (float): reward decay as defined in standard RL terminology
    """
    __slots__ = ["num_actions", "reward_decay"]

    def __init__(self, num_actions: int, reward_decay: float):
        self.num_actions = num_actions
        self.reward_decay = reward_decay


class PolicyGradient(AbsAlgorithm):
    """Policy gradient algorithm.

    The policy gradient algorithm base on the policy gradient theorem, a.k.a. REINFORCE.

    Args:
        policy_model (nn.Module): model for generating actions given states.
        optimizer_cls: torch optimizer class for the policy model.
        optimizer_params: parameters required for the policy optimizer class.
        hyper_params: hyper-parameter set for the AC algorithm.
    """

    def __init__(self, policy_model: nn.Module, optimizer_cls, optimizer_params,
                 hyper_params: PolicyGradientHyperParameters):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._policy_model = policy_model.to(self._device)
        self._policy_optimizer = optimizer_cls(self._policy_model.parameters(), **optimizer_params)
        self._hyper_params = hyper_params

    @property
    def model(self):
        return self._policy_model

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        state = torch.from_numpy(state).unsqueeze(0).to(self._device)   # (1, state_dim)
        action_dist = self._policy_model(state).squeeze()  # (num_actions,)
        return np.random.choice(self._hyper_params.num_actions, p=action_dist.numpy())

    def train(self, state_sequence: np.ndarray, action_sequence: np.ndarray, reward_sequence: np.ndarray):
        states = torch.from_numpy(state_sequence).to(self._device)   # (N, state_dim)
        returns = get_k_step_returns(reward_sequence, self._hyper_params.reward_decay)
        actions = torch.from_numpy(action_sequence).to(self._device)  # (N,)
        action_prob = self._policy_model(states).gather(1, actions.unsqueeze(1)).squeeze()   # (N, 1)
        policy_loss = -(torch.log(action_prob) * returns).mean()
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

    def load_trainable_models(self, policy_model):
        self._policy_model = policy_model

    def dump_trainable_models(self):
        return self._policy_model

    def load_trainable_models_from_file(self, path):
        """Load trainable models from disk."""
        self._policy_model = torch.load(path)

    def dump_trainable_models_to_file(self, path: str):
        """Dump the algorithm's trainable models to disk."""
        torch.save(self._policy_model.state_dict(), path)
