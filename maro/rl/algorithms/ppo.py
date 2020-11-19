# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import Callable

import numpy as np
import torch

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.models.learning_model import LearningModel
from maro.rl.utils.trajectory_utils import get_lambda_returns

from .utils import ActionWithLogProbability, expand_dim, preprocess, to_device, validate_task_names


class PPOTask(Enum):
    ACTOR = "actor"
    CRITIC = "critic"


class PPOConfig:
    """Configuration for the Proximal Policy Optimization (PPO) algorithm.

    Args:
        reward_decay (float): Reward decay as defined in standard RL terminology.
        critic_loss_func (Callable): Critic loss function.
        clip_ratio (float): Clip ratio as defined in PPO's objective function.
        actor_train_iters (int): Number of gradient descent steps for the policy model per call to ``train``.
        critic_train_iters (int): Number of gradient descent steps for the value model per call to ``train``.
        k (int): Number of time steps used in computing returns or return estimates. Defaults to -1, in which case
            rewards are accumulated until the end of the trajectory.
        lam (float): Lambda coefficient used in computing lambda returns. Defaults to 1.0, in which case the usual
            k-step return is computed.
    """
    __slots__ = [
        "reward_decay", "critic_loss_func", "clip_ratio", "actor_train_iters", "critic_train_iters", "k", "lam"
    ]

    def __init__(
        self,
        reward_decay: float,
        critic_loss_func: Callable,
        clip_ratio: float,
        actor_train_iters: int,
        critic_train_iters: int,
        k: int = -1,
        lam: float = 1.0
    ):
        self.reward_decay = reward_decay
        self.critic_loss_func = critic_loss_func
        self.clip_ratio = clip_ratio
        self.actor_train_iters = actor_train_iters
        self.critic_train_iters = critic_train_iters
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
    @validate_task_names(PPOTask)
    @to_device
    def __init__(self, model: LearningModel, config: PPOConfig):
        super().__init__(model, config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    @expand_dim
    def choose_action(self, state: np.ndarray):
        """Use the actor (policy) model to generate a stochastic action.

        Args:
            state: Input to the actor model.

        Returns:
            A ActionWithLogProbability namedtuple instance containing the action index and the corresponding
            log probability.
        """
        action_distribution = self._model(state, task_name="actor", is_training=False).squeeze().numpy()
        action = np.random.choice(len(action_distribution), p=action_distribution)
        return ActionWithLogProbability(action=action, log_probability=np.log(action_distribution[action]))

    def _get_values_and_bootstrapped_returns(self, states: torch.tensor, rewards: np.ndarray):
        state_values = self._model(states, task_name="critic").detach().squeeze()
        return_est = get_lambda_returns(
            rewards, state_values, self._config.reward_decay, self._config.lam, k=self._config.k
        )
        return state_values, return_est

    @preprocess
    def train(
        self, states: np.ndarray, actions: np.ndarray, log_action_prob_old: np.ndarray, rewards: np.ndarray
    ):
        state_values, return_est = self._get_values_and_bootstrapped_returns(states, rewards)
        advantages = return_est - state_values

        if self._model.shared_module and self._model.shared_module.is_trainable:
            pass
        else:
            # policy model training (with the value model fixed)
            for _ in range(self._config.actor_train_iters):
                action_prob = self._model(states, task_name="actor").gather(1, actions.unsqueeze(1)).squeeze()  # (N, 1)
                ratio = torch.exp(torch.log(action_prob) - log_action_prob_old)
                clipped_ratio = torch.clamp(ratio, 1 - self._config.clip_ratio, 1 + self._config.clip_ratio)
                actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
                self._model.learn(actor_loss)

            # value model training
            for _ in range(self._config.critic_train_iters):
                critic_loss = self._config.critic_loss_func(
                    self._model(states, task_name="critic").squeeze(), return_est
                )
                self._model.learn(critic_loss)
