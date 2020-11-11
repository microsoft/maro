# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import Callable

import numpy as np
import torch

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.models.learning_model import MultiTaskLearningModel
from maro.rl.utils.trajectory_utils import get_lambda_returns

from.task_validator import validate_tasks


class ActorCriticTask(Enum):
    ACTOR = "actor"
    CRITIC = "critic"


class ActorCriticConfig:
    """Configuration for the Actor-Critic algorithm.

    Args:
        num_actions (int): Number of possible actions
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
        "num_actions", "reward_decay", "critic_loss_func", "actor_train_iters", "critic_train_iters", "k", "lam"
    ]

    def __init__(
        self,
        num_actions: int,
        reward_decay: float,
        critic_loss_func: Callable,
        actor_train_iters: int,
        critic_train_iters: int,
        k: int = -1,
        lam: float = 1.0
    ):
        self.num_actions = num_actions
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
        core_model (MultiTaskLearningModel): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        config: Configuration for the AC algorithm.
    """
    @validate_tasks(ActorCriticTask)
    def __init__(self, core_model: MultiTaskLearningModel, config: ActorCriticConfig):
        super().__init__(core_model, config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._core_model.to(self._device)

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        state = torch.from_numpy(state).unsqueeze(0).to(self._device)   # (1, state_dim)
        self._core_model.eval()
        with torch.no_grad():
            action_dist = self._core_model(state, task="actor").squeeze().numpy()  # (num_actions,)
        return np.random.choice(self._config.num_actions, p=action_dist)

    def _get_values_and_bootstrapped_returns(self, state_sequence, reward_sequence):
        state_values = self._core_model(state_sequence, task="critic").detach().squeeze()
        state_values_numpy = state_values.numpy()
        return_est = get_lambda_returns(
            reward_sequence, state_values_numpy, self._config.reward_decay, self._config.lam,
            k=self._config.k
        )
        return_est = torch.from_numpy(return_est)
        return state_values, return_est

    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        states = torch.from_numpy(states).to(self._device)
        state_values, return_est = self._get_values_and_bootstrapped_returns(states, rewards)
        advantages = return_est - state_values
        actions = torch.from_numpy(actions).to(self._device)
        if self._core_model.has_trainable_shared_layers:
            pass
        else:
            # policy model training
            for _ in range(self._config.actor_train_iters):
                action_prob = self._core_model(states, task="actor").gather(1, actions.unsqueeze(1)).squeeze()  # (N,)
                actor_loss = -(torch.log(action_prob) * advantages).mean()
                self._core_model.step(actor_loss)

            # value model training
            for _ in range(self._config.critic_train_iters):
                critic_loss = self._config.critic_loss_func(
                    self._core_model(states, task="critic").squeeze(), return_est
                )
                self._core_model.step(critic_loss)
