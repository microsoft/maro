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
        policy_target_update_frequency (int): Number of training rounds between policy target model updates.
        q_value_target_update_frequency (int): Number of training rounds between policy target model updates.
        policy_tau (float): Soft update coefficient for the policy model, e.g.,
            target_model = tau * eval_model + (1-tau) * target_model
        q_value_tau (float): Soft update coefficient for the q-value model.
    """
    __slots__ = [
        "reward_decay", "q_value_loss_func", "policy_target_update_frequency", "q_value_target_update_frequency",
        "policy_tau", "q_value_tau"]

    def __init__(
        self,
        reward_decay: float,
        q_value_loss_func: Callable,
        policy_target_update_frequency: int,
        q_value_target_update_frequency: int,
        policy_tau: float = 1.0,
        q_value_tau: float = 1.0
    ):
        self.reward_decay = reward_decay
        self.q_value_loss_func = q_value_loss_func
        self.policy_target_update_frequency = policy_target_update_frequency
        self.q_value_target_update_frequency = q_value_target_update_frequency
        self.policy_tau = policy_tau
        self.q_value_tau = q_value_tau


class DDPG(AbsAlgorithm):
    """The Deep-Q-Networks algorithm.

    See https://arxiv.org/pdf/1509.02971.pdf for details.

    Args:
        model (LearningModel): Q-value model.
        config: Configuration for DQN algorithm.
    """
    @validate_task_names(DDPGTask)
    @to_device
    def __init__(
        self,
        model: LearningModel,
        config: DDPGConfig
    ):
        super().__init__(model, config)
        self._target_model = model.copy() if model.is_trainable else None
        self._policy_train_cnt = 0
        self._q_value_train_cnt = 0

    @expand_dim
    def choose_action(self, state):
        return self.model(state, task_name="actor", is_training=False)

    def _train_value_model(
        self, states: torch.tensor, actions: torch.tensor, rewards: torch.tensor, next_states: torch.tensor
    ):
        # value model training
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)  # (N, 1)

        current_q_values = self._model(torch.cat([states, actions]), task_name="q_value").squeeze(1)  # (N,)
        next_actions = self._model_dict["policy_target"](states).unsqueeze(dim=1)
        next_q_values = self._model_dict["value_target"](torch.cat([next_states, next_actions])).squeeze(1)  # (N,)
        target_q_values = (rewards + self._config.reward_decay * next_q_values).detach()  # (N,)
        loss = self._value_loss_func(current_q_values, target_q_values)
        self._model_dict["value"].train()
        self._value_optimizer.zero_grad()
        loss.backward()
        self._value_optimizer.step()
        self._value_train_cnt += 1
        if self._value_train_cnt % self._config.value_target_update_frequency == 0:
            self._update_target_model("value")

    def _train_policy_model(self, states: torch.tensor):
        # policy model training
        if hasattr(self, "_policy_optimizer"):
            loss = -self._model_dict["value"](torch.cat([states, self._model_dict["policy"](states)])).mean()
            self._model_dict["policy"].train()
            self._policy_optimizer.zero_grad()
            loss.backward()
            self._policy_optimizer.step()
            self._policy_train_cnt += 1
            if self._policy_train_cnt % self._config.policy_target_update_frequency == 0:
                self._update_target_model("policy")

    @preprocess
    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
        self._train_value_model(states, actions, rewards, next_states)
        self._train_policy_model(states)

    def _update_target_model(self, which: str):
        if which not in {"policy", "value"}:
            raise ValueError(f"unrecognized member: {which}")
        if hasattr(self, f"_{which}_optimizer"):
            tau = getattr(self._config, f"{which}_tau")
            for eval_params, target_params in zip(
                self._model_dict[which].parameters(), self._model_dict[f"{which}_target"].parameters()
            ):
                target_params.data = tau * eval_params.data + (1 - tau) * target_params.data

