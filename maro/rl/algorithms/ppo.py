# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import Callable

import numpy as np
import torch

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.models.learning_model import LearningModel
from maro.rl.utils.trajectory_utils import get_lambda_returns

from .utils import expand_dim, preprocess, to_device, validate_task_names


class PPOTask(Enum):
    ACTOR = "actor"
    CRITIC = "critic"


class PPOConfig:
    """Configuration for the Proximal Policy Optimization (PPO) algorithm.

    Args:
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
        "reward_decay", "critic_loss_func", "clip_ratio", "policy_train_iters", "value_train_iters", "k", "lam"
    ]

    def __init__(
        self,
        reward_decay: float,
        critic_loss_func: Callable,
        clip_ratio: float,
        policy_train_iters: int,
        value_train_iters: int,
        k: int = -1,
        lam: float = 1.0
    ):
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
        model (LearningModel): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        config: Configuration for the PPO algorithm.
    """
    @to_device
    @validate_task_names(PPOTask)
    def __init__(self, model: LearningModel, config: PPOConfig):
        super().__init__(model, config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    @expand_dim
    def choose_action(self, state: np.ndarray):
        action_distribution = self._model(state, task_name="actor", is_training=False).squeeze().numpy()
        return np.random.choice(len(action_distribution), p=action_distribution)

    def _get_values_and_bootstrapped_returns(self, states: torch.tensor, rewards: np.ndarray):
        state_values = self._model(states, task_name="critic").detach().squeeze()
        return_est = get_lambda_returns(
            rewards, state_values, self._config.reward_decay, self._config.lam, k=self._config.k
        )
        return state_values, return_est

    @preprocess
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
            action_prob = self._model(states, task_name="actor").gather(1, actions.unsqueeze(1)).squeeze()  # (N, 1)
            ratio = torch.exp(torch.log(action_prob) - log_action_prob_old)
            clipped_ratio = torch.clamp(ratio, 1 - self._config.clip_ratio, 1 + self._config.clip_ratio)
            loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
            self._model.learn(loss)

        # value model training
        for _ in range(self._config.value_train_iters):
            loss = self._config.lovalue_loss_func(self._model(states, task_name="critic"), return_est)
            self._model.learn(loss)
