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


class ActorCriticTask(Enum):
    ACTOR = "actor"
    CRITIC = "critic"


class ActorCriticConfig:
    """Configuration for the Actor-Critic algorithm.

    Args:
        reward_decay (float): Reward decay as defined in standard RL terminology.
        critic_loss_func (Callable): Loss function for the critic model.
        actor_train_iters (int): Number of gradient descent steps for the policy model per call to ``train``.
        critic_train_iters (int): Number of gradient descent steps for the value model per call to ``train``.
        k (int): Number of time steps used in computing returns or return estimates. Defaults to -1, in which case
            rewards are accumulated until the end of the trajectory.
        lam (float): Lambda coefficient used in computing lambda returns. Defaults to 1.0, in which case the usual
            k-step return is computed.
    """
    __slots__ = [
        "reward_decay", "critic_loss_func", "actor_train_iters", "critic_train_iters", "k", "lam"
    ]

    def __init__(
        self,
        reward_decay: float,
        critic_loss_func: Callable,
        actor_train_iters: int,
        critic_train_iters: int,
        k: int = -1,
        lam: float = 1.0
    ):
        self.reward_decay = reward_decay
        self.critic_loss_func = critic_loss_func
        self.actor_train_iters = actor_train_iters
        self.critic_train_iters = critic_train_iters
        self.k = k
        self.lam = lam


class ActorCritic(AbsAlgorithm):
    """Actor Critic algorithm with separate policy and value models (no shared layers).

    The Actor-Critic algorithm base on the policy gradient theorem.

    Args:
        model (LearningModel): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        config: Configuration for the AC algorithm.
    """
    @to_device
    @validate_task_names(ActorCriticTask)
    def __init__(self, model: LearningModel, config: ActorCriticConfig):
        super().__init__(model, config)

    @expand_dim
    def choose_action(self, state: np.ndarray):
        action_distribution = self._model(state, task_name="actor", is_training=False).squeeze().numpy()
        return np.random.choice(len(action_distribution), p=action_distribution)

    def _get_values_and_bootstrapped_returns(self, state_sequence, reward_sequence):
        state_values = self._model(state_sequence, task_name="critic").detach().squeeze()
        return_est = get_lambda_returns(
            reward_sequence, state_values, self._config.reward_decay, self._config.lam, k=self._config.k
        )
        return state_values, return_est

    @preprocess
    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        state_values, return_est = self._get_values_and_bootstrapped_returns(states, rewards)
        advantages = return_est - state_values
        if self._model.shared_module and self._model.shared_module.is_trainable:
            pass
        else:
            # policy model training
            for _ in range(self._config.actor_train_iters):
                action_prob = self._model(states, task_name="actor").gather(1, actions.unsqueeze(1)).squeeze()  # (N,)
                actor_loss = -(torch.log(action_prob) * advantages).mean()
                self._model.learn(actor_loss)

            # value model training
            for _ in range(self._config.critic_train_iters):
                critic_loss = self._config.critic_loss_func(
                    self._model(states, task_name="critic").squeeze(), return_est
                )
                self._model.learn(critic_loss)
