# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import Callable

import numpy as np
import torch

from maro.rl.models.learning_model import LearningModel

from .abs_algorithm import AbsAlgorithm
from .utils import expand_dim, preprocess, to_device, validate_task_names


class DDPGTask(Enum):
    POLICY = "policy"
    Q_VALUE = "q_value"


class DDPGConfig:
    """Configuration for the DDPG algorithm.
    Args:
        reward_decay (float): Reward decay as defined in standard RL terminology
        q_value_loss_func (Callable):
        target_update_frequency (int): Number of training rounds between policy target model updates.
        tau (float): Soft update coefficient, e.g., target_model = tau * eval_model + (1-tau) * target_model
    """
    __slots__ = ["reward_decay", "q_value_loss_func", "target_update_frequency", "tau"]

    def __init__(
        self,
        reward_decay: float,
        q_value_loss_func: Callable,
        target_update_frequency: int,
        tau: float = 1.0
    ):
        self.reward_decay = reward_decay
        self.q_value_loss_func = q_value_loss_func
        self.target_update_frequency = target_update_frequency
        self.tau = tau


class DDPG(AbsAlgorithm):
    """The Deep-Q-Networks algorithm.

    See https://arxiv.org/pdf/1509.02971.pdf for details.

    Args:
        model (LearningModel): DDPG policy and q-value models.
        config: Configuration for DDPG algorithm.
    """
    @validate_task_names(DDPGTask)
    @to_device
    def __init__(self, model: LearningModel, config: DDPGConfig):
        super().__init__(model, config)
        self._target_model = model.copy() if model.is_trainable else None
        self._train_cnt = 0

    @expand_dim
    def choose_action(self, state):
        return self.model(state, task_name="actor", is_training=False)

    @preprocess
    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
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
