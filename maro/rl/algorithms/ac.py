# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import Callable

import numpy as np
import torch

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.models.learning_model import LearningModuleManager
from maro.rl.utils.trajectory_utils import get_lambda_returns

from .utils import ActionWithLogProbability, expand_dim, preprocess, to_device, validate_task_names


class ActorCriticTask(Enum):
    ACTOR = "actor"
    CRITIC = "critic"


class ActorCriticConfig:
    """Configuration for the Actor-Critic algorithm.

    Args:
        reward_decay (float): Reward decay as defined in standard RL terminology.
        critic_loss_func (Callable): Loss function for the critic model.
        train_iters (int): Number of gradient descent steps per call to ``train``.
        actor_loss_coefficient (float): The coefficient for actor loss in the total loss function, e.g.,
            loss = critic_loss + ``actor_loss_coefficient`` * actor_loss. Defaults to 1.0.
        k (int): Number of time steps used in computing returns or return estimates. Defaults to -1, in which case
            rewards are accumulated until the end of the trajectory.
        lam (float): Lambda coefficient used in computing lambda returns. Defaults to 1.0, in which case the usual
            k-step return is computed.
        clip_ratio (float): Clip ratio in the PPO algorithm (https://arxiv.org/pdf/1707.06347.pdf). Defaults to None,
            in which case the actor loss is calculated using the usual policy gradient theorem.
    """
    __slots__ = [
        "reward_decay", "critic_loss_func", "train_iters", "actor_loss_coefficient", "k", "lam", "clip_ratio"
    ]

    def __init__(
        self,
        reward_decay: float,
        critic_loss_func: Callable,
        train_iters: int,
        actor_loss_coefficient: float = 1.0,
        k: int = -1,
        lam: float = 1.0,
        clip_ratio: float = None
    ):
        self.reward_decay = reward_decay
        self.critic_loss_func = critic_loss_func
        self.train_iters = train_iters
        self.actor_loss_coefficient = actor_loss_coefficient
        self.k = k
        self.lam = lam
        self.clip_ratio = clip_ratio


class ActorCritic(AbsAlgorithm):
    """Actor Critic algorithm with separate policy and value models (no shared layers).

    The Actor-Critic algorithm base on the policy gradient theorem.

    Args:
        model (LearningModuleManager): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        config: Configuration for the AC algorithm.
    """
    @validate_task_names(ActorCriticTask)
    @to_device
    def __init__(self, model: LearningModuleManager, config: ActorCriticConfig):
        super().__init__(model, config)

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

    def _get_values_and_bootstrapped_returns(self, state_sequence, reward_sequence):
        state_values = self._model(state_sequence, task_name="critic").detach().squeeze()
        return_est = get_lambda_returns(
            reward_sequence, state_values, self._config.reward_decay, self._config.lam, k=self._config.k
        )
        return state_values, return_est

    @preprocess
    def train(
        self, states: np.ndarray, actions: np.ndarray, log_action_prob: np.ndarray, rewards: np.ndarray
    ):
        state_values, return_est = self._get_values_and_bootstrapped_returns(states, rewards)
        advantages = return_est - state_values
        for _ in range(self._config.train_iters):
            critic_loss = self._config.critic_loss_func(
                self._model(states, task_name="critic").squeeze(), return_est
            )
            action_prob = self._model(states, task_name="actor").gather(1, actions.unsqueeze(1)).squeeze()  # (N,)
            log_action_prob_new = torch.log(action_prob)
            actor_loss = self._actor_loss(log_action_prob_new, log_action_prob, advantages)
            loss = critic_loss + self._config.actor_loss_coefficient * actor_loss
            self._model.learn(loss)

    def _actor_loss(self, log_action_prob_new, log_action_prob_old, advantages):
        if self._config.clip_ratio is not None:
            ratio = torch.exp(log_action_prob_new - log_action_prob_old)
            clip_ratio = torch.clamp(ratio, 1 - self._config.clip_ratio, 1 + self._config.clip_ratio)
            actor_loss = -(torch.min(ratio * advantages, clip_ratio * advantages)).mean()
        else:
            actor_loss = -(log_action_prob_new * advantages).mean()

        return actor_loss
