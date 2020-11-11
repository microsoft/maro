# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.models.learning_model import MultiTaskLearningModel
from maro.rl.utils.trajectory_utils import get_lambda_returns

from .task_validator import validate_tasks


class PPOTask(Enum):
    ACTOR = "actor"
    CRITIC = "critic"


class PPOConfig:
    """Configuration for the Proximal Policy Optimization (PPO) algorithm.

    Args:
        num_actions (int): Number of possible actions.
        reward_decay (float): Reward decay as defined in standard RL terminology.
        critic_loss_func (Callable): Critic loss function.
        clip_ratio (float): Clip ratio as defined in PPO's objective function.
        policy_train_iters (int): Number of gradient descent steps for the policy model per call to ``train``.
        value_train_iters (int): Number of gradient descent steps for the value model per call to ``train``.
        k (int): Number of time steps used in computing returns or return estimates. Defaults to -1, in which case
            rewards are accumulated until the end of the trajectory.
        lam (float): Lambda coefficient used in computing lambda returns. Defaults to 1.0, in which case the usual
            k-step return is computed.
    """
    __slots__ = [
        "num_actions", "reward_decay", "critic_loss_func", "clip_ratio", "policy_train_iters", "value_train_iters",
        "k", "lam"
    ]

    def __init__(
        self,
        num_actions: int,
        reward_decay: float,
        critic_loss_func: Callable,
        clip_ratio: float,
        policy_train_iters: int,
        value_train_iters: int,
        k: int = -1,
        lam: float = 1.0
    ):
        self.num_actions = num_actions
        self.reward_decay = reward_decay
        self.critic_loss_func = critic_loss_func
        self.clip_ratio = clip_ratio
        self.policy_train_iters = policy_train_iters
        self.value_train_iters = value_train_iters
        self.k = k
        self.lam = lam


class PPO(AbsAlgorithm):
    """Proximal Policy Optimization (PPO) algorithm.

    See https://arxiv.org/pdf/1707.06347.pdf for details.

    Args:
        core_model (MultiTaskLearningModel): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        config: Configuration for the PPO algorithm.
    """
    @validate_tasks(PPOTask)
    def __init__(self, core_model: MultiTaskLearningModel, config: PPOConfig):
        super().__init__(core_model, config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._core_model.to(self._device)

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        state = torch.from_numpy(state).unsqueeze(0).to(self._device)   # (1, state_dim)
        self._core_model.eval()
        with torch.no_grad():
            action_dist = self._core_model(state, task="actor").squeeze().numpy()  # (num_actions,)
        return np.random.choice(self._config.num_actions, p=action_dist)

    def _get_values_and_bootstrapped_returns(self, states: torch.tensor, rewards: np.ndarray):
        state_values = self._core_model(states, task="critic").detach().squeeze()
        state_values_numpy = state_values.numpy()
        return_est = get_lambda_returns(
            rewards, state_values_numpy, self._config.reward_decay, self._config.lam,
            k=self._config.k
        )
        return_est = torch.from_numpy(return_est)
        return state_values, return_est

    def train(
        self, states: np.ndarray, actions: np.ndarray, log_action_prob: np.ndarray, rewards: np.ndarray
    ):
        states = torch.from_numpy(states).to(self._device)  # (N, state_dim)
        state_values, return_est = self._get_values_and_bootstrapped_returns(states, rewards)
        advantages = return_est - state_values
        actions = torch.from_numpy(actions).to(self._device)  # (N,)
        log_action_prob_old = torch.from_numpy(log_action_prob).to(self._device)

        # policy model training (with the value model fixed)
        for _ in range(self._config.policy_train_iters):
            action_prob = self._core_model(states, task="actor").gather(1, actions.unsqueeze(1)).squeeze()  # (N, 1)
            ratio = torch.exp(torch.log(action_prob) - log_action_prob_old)
            clipped_ratio = torch.clamp(ratio, 1 - self._config.clip_ratio, 1 + self._config.clip_ratio)
            loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
            self._core_model.step(loss)

        # value model training
        for _ in range(self._config.value_train_iters):
            loss = self._config.lovalue_loss_func(self._core_model(states, task="critic"), return_est)
            self._core_model.step(loss)
