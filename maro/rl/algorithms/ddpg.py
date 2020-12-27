# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

import numpy as np
import torch

from maro.rl.models.learning_model import LearningModuleManager

from .abs_algorithm import AbsAlgorithm


class DDPGConfig:
    """Configuration for the DDPG algorithm.
    Args:
        reward_decay (float): Reward decay as defined in standard RL terminology
        q_value_loss_func (Callable): Loss function for the Q-value estimator.
        target_update_frequency (int): Number of training rounds between policy target model updates.
        explorer_cls: Explorer class.
        explorer_params: Parameters required for the explorer class.
        tau (float): Soft update coefficient, e.g., target_model = tau * eval_model + (1-tau) * target_model.
            Defaults to 1.0.
    """
    __slots__ = [
        "reward_decay", "q_value_loss_func", "target_update_frequency", "tau", "explorer_cls", "explorer_params"
    ]

    def __init__(
        self,
        reward_decay: float,
        q_value_loss_func: Callable,
        target_update_frequency: int,
        explorer_cls,
        explorer_params: dict,
        tau: float = 1.0,
    ):
        self.reward_decay = reward_decay
        self.q_value_loss_func = q_value_loss_func
        self.target_update_frequency = target_update_frequency
        self.explorer_cls = explorer_cls
        self.explorer_params = explorer_params
        self.tau = tau


class DDPG(AbsAlgorithm):
    """The Deep-Q-Networks algorithm.

    See https://arxiv.org/pdf/1509.02971.pdf for details.

    Args:
        model (LearningModel): DDPG policy and q-value models.
        config: Configuration for DDPG algorithm.
    """
    def __init__(self, model: LearningModuleManager, config: DDPGConfig):
        self.validate_task_names(model.task_names, {"policy", "q_value"})
        super().__init__(model, config)
        self._explorer = self._config.explorer_cls(self._model.output_dim["policy"], **self._config.explorer_params)
        self._target_model = model.copy() if model.is_trainable else None
        self._train_cnt = 0

    def choose_action(self, state) -> np.ndarray:
        state = torch.from_numpy(state).to(self._device)
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(dim=0)

        model_action = self.model(state, task_name="policy", is_training=False)
        if is_single:
            return self._explorer(model_action)

        # batch inference
        return np.vstack([self._explorer(action) for action in model_action])

    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
        states = torch.from_numpy(states).to(self._device)
        actions = torch.from_numpy(actions).to(self._device)
        rewards = torch.from_numpy(rewards).to(self._device)
        next_states = torch.from_numpy(next_states).to(self._device)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)  # (N, 1)
        current_q_values = self._model(torch.cat([states, actions]), task_name="q_value").squeeze(1)  # (N,)
        next_actions = self._target_model(states, task_name="policy", is_training=False).unsqueeze(dim=1)
        next_q_values = self._target_model(
            torch.cat([next_states, next_actions]), task_name="q_value", is_training=False
        ).squeeze(1)  # (N,)
        target_q_values = (rewards + self._config.reward_decay * next_q_values).detach()  # (N,)
        q_value_loss = self._config.q_value_loss_func(current_q_values, target_q_values)
        policy_loss = -self._model(torch.cat([states, self._model(states, task_name="policy")])).mean()
        self._model.learn(q_value_loss + policy_loss)
        self._train_cnt += 1
        if self._train_cnt % self._config.target_update_frequency == 0:
            self._target_model.soft_update(self._model, self._config.tau)
