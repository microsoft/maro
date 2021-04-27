# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Union

import numpy as np
import torch

from maro.rl.exploration import NoiseExplorer
from maro.rl.model import SimpleMultiHeadModel
from maro.utils.exception.rl_toolkit_exception import UnrecognizedTask

from .abs_agent import AbsAgent


class DDPGConfig:
    """Configuration for the DDPG algorithm.
    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        q_value_loss_func (Callable): Loss function for the Q-value estimator.
        target_update_freq (int): Number of training rounds between policy target model updates.
        actor_loss_coefficient (float): The coefficient for policy loss in the total loss function, e.g.,
            loss = q_value_loss + ``policy_loss_coefficient`` * policy_loss. Defaults to 1.0.
        tau (float): Soft update coefficient, e.g., target_model = tau * eval_model + (1-tau) * target_model.
            Defaults to 1.0.
    """
    __slots__ = ["reward_discount", "q_value_loss_func", "target_update_freq", "policy_loss_coefficient", "tau"]

    def __init__(
        self,
        reward_discount: float,
        q_value_loss_func: Callable,
        target_update_freq: int,
        policy_loss_coefficient: float = 1.0,
        tau: float = 1.0,
    ):
        self.reward_discount = reward_discount
        self.q_value_loss_func = q_value_loss_func
        self.target_update_freq = target_update_freq
        self.policy_loss_coefficient = policy_loss_coefficient
        self.tau = tau


class DDPG(AbsAgent):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
    https://arxiv.org/pdf/1509.02971.pdf
    https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg

    Args:
        model (SimpleMultiHeadModel): DDPG policy and q-value models.
        config: Configuration for DDPG algorithm.
        explorer (NoiseExplorer): An NoiseExplorer instance for generating exploratory actions. Defaults to None.
    """
    def __init__(self, model: SimpleMultiHeadModel, config: DDPGConfig, explorer: NoiseExplorer = None):
        if model.task_names is None or set(model.task_names) != {"policy", "q_value"}:
            raise UnrecognizedTask(f"Expected model task names 'policy' and 'q_value', but got {model.task_names}")
        super().__init__(model, config)
        self._explorer = explorer
        self._target_model = model.copy() if model.trainable else None
        self._train_cnt = 0

    def choose_action(self, state) -> Union[float, np.ndarray]:
        state = torch.from_numpy(state).to(self.device)
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(dim=0)

        action = self.model(state, task_name="policy", training=False).data.cpu().numpy()
        action_dim = action.shape[1]
        if self._explorer:
            action = self._explorer(action)

        if action_dim == 1:
            action = action.squeeze(axis=1)

        return action[0] if is_single else action

    def learn(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
        states = torch.from_numpy(states).to(self.device)
        actual_actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        if len(actual_actions.shape) == 1:
            actual_actions = actual_actions.unsqueeze(dim=1)  # (N, 1)

        current_q_values = self.model(torch.cat([states, actual_actions], dim=1), task_name="q_value")
        current_q_values = current_q_values.squeeze(dim=1)  # (N,)
        next_actions = self._target_model(states, task_name="policy", training=False)
        next_q_values = self._target_model(
            torch.cat([next_states, next_actions], dim=1), task_name="q_value", training=False
        ).squeeze(1)  # (N,)
        target_q_values = (rewards + self.config.reward_discount * next_q_values).detach()  # (N,)
        q_value_loss = self.config.q_value_loss_func(current_q_values, target_q_values)
        actions_from_model = self.model(states, task_name="policy")
        policy_loss = -self.model(torch.cat([states, actions_from_model], dim=1), task_name="q_value").mean()
        self.model.learn(q_value_loss + self.config.policy_loss_coefficient * policy_loss)
        self._train_cnt += 1
        if self._train_cnt % self.config.target_update_freq == 0:
            self._target_model.soft_update(self.model, self.config.tau)
