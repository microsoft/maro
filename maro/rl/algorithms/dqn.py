# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum

import numpy as np

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.models.learning_model import LearningModel

from .utils import expand_dim, preprocess, to_device, validate_task_names


class DuelingDQNTask(Enum):
    STATE_VALUE = "state_value"
    ADVANTAGE = "advantage"


class DQNConfig:
    """Configuration for the DQN algorithm.

    Args:
        num_actions (int): Number of possible actions.
        reward_decay (float): Reward decay as defined in standard RL terminology.
        loss_cls: Loss function class for evaluating TD errors.
        target_update_frequency (int): Number of training rounds between target model updates.
        tau (float): Soft update coefficient, i.e., target_model = tau * eval_model + (1 - tau) * target_model.
        is_double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
        advantage_mode (str): Advantage mode for the dueling architecture. Defaults to None, in which
            case it is assumed that the regular Q-value model is used.
        per_sample_td_error_enabled (bool): If True, per-sample TD errors will be returned by the DQN's train()
            method. Defaults to False.
    """
    __slots__ = [
        "num_actions", "reward_decay", "loss_func", "target_update_frequency", "tau", "is_double",
        "advantage_mode", "per_sample_td_error_enabled"
    ]

    def __init__(
        self,
        num_actions: int,
        reward_decay: float,
        loss_cls,
        target_update_frequency: int,
        tau: float = 0.1,
        is_double: bool = True,
        advantage_mode: str = None,
        per_sample_td_error_enabled: bool = False
    ):
        self.num_actions = num_actions
        self.reward_decay = reward_decay
        self.target_update_frequency = target_update_frequency
        self.tau = tau
        self.is_double = is_double
        self.advantage_mode = advantage_mode
        self.per_sample_td_error_enabled = per_sample_td_error_enabled
        self.loss_func = loss_cls(reduction="none" if per_sample_td_error_enabled else "mean")


class DQN(AbsAlgorithm):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        model (LearningModel): Q-value model.
        config: Configuration for DQN algorithm.
    """
    @to_device
    @validate_task_names(DuelingDQNTask)
    def __init__(self, model: LearningModel, config: DQNConfig):
        super().__init__(model, config)
        self._training_counter = 0
        self._target_model = model.copy() if model.is_trainable else None

    @expand_dim
    def choose_action(self, state: np.ndarray, epsilon=None):
        if epsilon is None or np.random.rand() > epsilon:
            q_values = self._get_q_values(self._model, state, is_training=False)
            return q_values.argmax(dim=1).item()

        return np.random.choice(self._config.num_actions)

    def _get_q_values(self, model, states, is_training: bool = True):
        if self._config.advantage_mode is not None:
            output = model(states, is_training=is_training)
            state_values = output["state_value"]
            advantages = output["advantage"]
            # Use mean or max correction to address the identifiability issue
            corrections = advantages.mean(1) if self._config.advantage_mode == "mean" else advantages.max(1)[0]
            q_values = state_values + advantages - corrections.unsqueeze(1)
            return q_values
        else:
            return model(states, is_training=is_training)

    def _get_next_q_values(self, current_q_values_for_all_actions, next_states):
        next_q_values_for_all_actions = self._get_q_values(self._target_model, next_states, is_training=False)
        if self._config.is_double:
            actions = current_q_values_for_all_actions.max(dim=1)[1].unsqueeze(1)
            return next_q_values_for_all_actions.gather(1, actions).squeeze(1)  # (N,)
        else:
            return next_q_values_for_all_actions.max(dim=1)[0]   # (N,)

    def _compute_td_errors(self, states, actions, rewards, next_states):
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)  # (N, 1)
        current_q_values_for_all_actions = self._get_q_values(self._model, states)
        current_q_values = current_q_values_for_all_actions.gather(1, actions).squeeze(1)  # (N,)
        next_q_values = self._get_next_q_values(current_q_values_for_all_actions, next_states)  # (N,)
        target_q_values = (rewards + self._config.reward_decay * next_q_values).detach()  # (N,)
        return self._config.loss_func(current_q_values, target_q_values)

    @preprocess
    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
        loss = self._compute_td_errors(states, actions, rewards, next_states)
        self._model.learn(loss.mean() if self._config.per_sample_td_error_enabled else loss)
        self._training_counter += 1
        if self._training_counter % self._config.target_update_frequency == 0:
            self._update_targets()

        return loss.detach().numpy()

    def _update_targets(self):
        for eval_params, target_params in zip(
            self._model.parameters(), self._target_model.parameters()
        ):
            target_params.data = (
                self._config.tau * eval_params.data + (1 - self._config.tau) * target_params.data
            )
