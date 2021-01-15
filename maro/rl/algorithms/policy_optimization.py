# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple
from typing import Callable, List, Union

import numpy as np
import torch

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.models.learning_model import LearningModel
from maro.rl.utils.trajectory_utils import get_lambda_returns, get_truncated_cumulative_reward

ActionInfo = namedtuple("ActionInfo", ["action", "log_probability"])


class PolicyOptimizationConfig:
    """Configuration for the policy optimization algorithm family."""
    __slots__ = ["reward_discount"]

    def __init__(self, reward_discount):
        self.reward_discount = reward_discount


class PolicyOptimization(AbsAlgorithm):
    """Policy optimization algorithm family.

    The algorithm family includes policy gradient (e.g. REINFORCE), actor-critic, PPO, etc.
    """
    def choose_action(self, state: np.ndarray) -> Union[ActionInfo, List[ActionInfo]]:
        """Use the actor (policy) model to generate stochastic actions.

        Args:
            state: Input to the actor model.

        Returns:
            A single ActionInfo namedtuple or a list of ActionInfo namedtuples.
        """
        state = torch.from_numpy(state).to(self._device)
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(dim=0)

        action_distribution = self._model(state, task_name="actor", is_training=False).squeeze().numpy()
        if is_single:
            action = np.random.choice(len(action_distribution), p=action_distribution)
            return ActionInfo(action=action, log_probability=np.log(action_distribution[action]))

        # batch inference
        batch_results = []
        for distribution in action_distribution:
            action = np.random.choice(len(distribution), p=distribution)
            batch_results.append(ActionInfo(action=action, log_probability=np.log(distribution[action])))

        return batch_results

    def train(
        self, states: np.ndarray, actions: np.ndarray, log_action_prob: np.ndarray, rewards: np.ndarray
    ):
        raise NotImplementedError


class PolicyGradient(PolicyOptimization):
    """The vanilla Policy Gradient (VPG) algorithm, a.k.a., REINFORCE.

    Reference: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.
    """
    def train(
        self, states: np.ndarray, actions: np.ndarray, log_action_prob: np.ndarray, rewards: np.ndarray
    ):
        states = torch.from_numpy(states).to(self._device)
        actions = torch.from_numpy(actions).to(self._device)
        returns = get_truncated_cumulative_reward(rewards, self._config.reward_discount)
        returns = torch.from_numpy(returns).to(self._device)
        action_distributions = self._model(states)
        action_prob = action_distributions.gather(1, actions.unsqueeze(1)).squeeze()   # (N, 1)
        loss = -(torch.log(action_prob) * returns).mean()
        self._model.learn(loss)


class ActorCriticConfig(PolicyOptimizationConfig):
    """Configuration for the Actor-Critic algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
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
        "reward_discount", "critic_loss_func", "train_iters", "actor_loss_coefficient", "k", "lam", "clip_ratio"
    ]

    def __init__(
        self,
        reward_discount: float,
        critic_loss_func: Callable,
        train_iters: int,
        actor_loss_coefficient: float = 1.0,
        k: int = -1,
        lam: float = 1.0,
        clip_ratio: float = None
    ):
        super().__init__(reward_discount)
        self.critic_loss_func = critic_loss_func
        self.train_iters = train_iters
        self.actor_loss_coefficient = actor_loss_coefficient
        self.k = k
        self.lam = lam
        self.clip_ratio = clip_ratio


class ActorCritic(PolicyOptimization):
    """Actor Critic algorithm with separate policy and value models.

    References:
    https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.
    https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

    Args:
        model (LearningModel): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        config: Configuration for the AC algorithm.
    """
    def __init__(self, model: LearningModel, config: ActorCriticConfig):
        self.validate_task_names(model.task_names, {"actor", "critic"})
        super().__init__(model, config)

    def _get_values_and_bootstrapped_returns(self, state_sequence, reward_sequence):
        state_values = self._model(state_sequence, task_name="critic").detach().squeeze()
        return_est = get_lambda_returns(
            reward_sequence, state_values, self._config.reward_discount, self._config.lam, k=self._config.k
        )
        return state_values, return_est

    def train(
        self, states: np.ndarray, actions: np.ndarray, log_action_prob: np.ndarray, rewards: np.ndarray
    ):
        states = torch.from_numpy(states).to(self._device)
        actions = torch.from_numpy(actions).to(self._device)
        log_action_prob = torch.from_numpy(log_action_prob).to(self._device)
        rewards = torch.from_numpy(rewards).to(self._device)
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
