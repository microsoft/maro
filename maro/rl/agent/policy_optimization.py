# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from maro.rl.model import SimpleMultiHeadModel
from maro.rl.utils.trajectory_utils import get_lambda_returns, get_truncated_cumulative_reward
from maro.utils.exception.rl_toolkit_exception import UnrecognizedTask

from .abs_agent import AbsAgent


class PolicyGradient(AbsAgent):
    """The vanilla Policy Gradient (VPG) algorithm, a.k.a., REINFORCE.

    Reference: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.

    Args:
        name (str): Agent's name.
        model (SimpleMultiHeadModel): Model that computes action distributions.
        reward_discount (float): Reward decay as defined in standard RL terminology.
    """
    def __init__(self, name: str, model: SimpleMultiHeadModel, reward_discount: float):
        super().__init__(name, model, reward_discount)

    def choose_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use the actor (policy) model to generate stochastic actions.

        Args:
            state: Input to the actor model.

        Returns:
            Actions and corresponding log probabilities.
        """
        state = torch.from_numpy(state).to(self._device)
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(dim=0)

        action_prob = Categorical(self._model(state, is_training=False))
        action = action_prob.sample()
        log_p = action_prob.log_prob(action)
        action, log_p = action.cpu().numpy(), log_p.cpu().numpy()
        return (action[0], log_p[0]) if is_single else (action, log_p)

    def train(
        self, states: np.ndarray, actions: np.ndarray, log_action_prob: np.ndarray, rewards: np.ndarray
    ):
        states = torch.from_numpy(states).to(self._device)
        actions = torch.from_numpy(actions).to(self._device)
        returns = get_truncated_cumulative_reward(rewards, self._config)
        returns = torch.from_numpy(returns).to(self._device)
        action_distributions = self._model(states)
        action_prob = action_distributions.gather(1, actions.unsqueeze(1)).squeeze()   # (N, 1)
        loss = -(torch.log(action_prob) * returns).mean()
        self._model.learn(loss)


class ActorCriticConfig:
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
        self.reward_discount = reward_discount
        self.critic_loss_func = critic_loss_func
        self.train_iters = train_iters
        self.actor_loss_coefficient = actor_loss_coefficient
        self.k = k
        self.lam = lam
        self.clip_ratio = clip_ratio


class ActorCritic(AbsAgent):
    """Actor Critic algorithm with separate policy and value models.

    References:
    https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.
    https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

    Args:
        name (str): Agent's name.
        model (SimpleMultiHeadModel): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        config: Configuration for the AC algorithm.
    """
    def __init__(self, name: str, model: SimpleMultiHeadModel, config: ActorCriticConfig):
        if model.task_names is None or set(model.task_names) != {"actor", "critic"}:
            raise UnrecognizedTask(f"Expected model task names 'actor' and 'critic', but got {model.task_names}")
        super().__init__(name, model, config)

    def choose_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use the actor (policy) model to generate stochastic actions.

        Args:
            state: Input to the actor model.

        Returns:
            Actions and corresponding log probabilities.
        """
        state = torch.from_numpy(state).to(self._device)
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(dim=0)

        action_prob = Categorical(self._model(state, task_name="actor", is_training=False))
        action = action_prob.sample()
        log_p = action_prob.log_prob(action)
        action, log_p = action.cpu().numpy(), log_p.cpu().numpy()
        return (action[0], log_p[0]) if is_single else (action, log_p)

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

    def _get_values_and_bootstrapped_returns(self, state_sequence, reward_sequence):
        state_values = self._model(state_sequence, task_name="critic").detach().squeeze()
        return_est = get_lambda_returns(
            reward_sequence, state_values, self._config.reward_discount, self._config.lam, k=self._config.k
        )
        return state_values, return_est
