# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np
import torch

from maro.rl.models.learning_model import LearningModel

from .abs_algorithm import AbsAlgorithm


class DQNConfig:
    """Configuration for the DQN algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        loss_cls: Loss function class for evaluating TD errors.
        target_update_frequency (int): Number of training rounds between target model updates.
        epsilon (float): Exploration rate for epsilon-greedy exploration. Defaults to None.
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
        "reward_discount", "loss_func", "target_update_frequency", "epsilon", "tau", "is_double", "advantage_mode",
        "per_sample_td_error_enabled"
    ]

    def __init__(
        self,
        reward_discount: float,
        loss_cls,
        target_update_frequency: int,
        epsilon: float = .0,
        tau: float = 0.1,
        is_double: bool = True,
        advantage_mode: str = None,
        per_sample_td_error_enabled: bool = False
    ):
        self.reward_discount = reward_discount
        self.target_update_frequency = target_update_frequency
        self.epsilon = epsilon
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
    def __init__(self, model: LearningModel, config: DQNConfig):
        self.validate_task_names(model.task_names, {"state_value", "advantage"})
        super().__init__(model, config)
        if isinstance(self._model.output_dim, int):
            self._num_actions = self._model.output_dim
        else:
            self._num_actions = self._model.output_dim["advantage"]
        self._training_counter = 0
        self._target_model = model.copy() if model.is_trainable else None

    def choose_action(self, state: np.ndarray) -> Union[int, np.ndarray]:
        state = torch.from_numpy(state).to(self._device)
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(dim=0)

        greedy_action = self._get_q_values(self._model, state, is_training=False).argmax(dim=1).data
        # No exploration
        if self._config.epsilon == .0:
            return greedy_action.item() if is_single else greedy_action.numpy()

        if is_single:
            return greedy_action if np.random.random() > self._config.epsilon else np.random.choice(self._num_actions)

        # batch inference
        return np.array([
            act if np.random.random() > self._config.epsilon else np.random.choice(self._num_actions)
            for act in greedy_action
        ])

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
        target_q_values = (rewards + self._config.reward_discount * next_q_values).detach()  # (N,)
        return self._config.loss_func(current_q_values, target_q_values)

    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
        states = torch.from_numpy(states).to(self._device)
        actions = torch.from_numpy(actions).to(self._device)
        rewards = torch.from_numpy(rewards).to(self._device)
        next_states = torch.from_numpy(next_states).to(self._device)
        loss = self._compute_td_errors(states, actions, rewards, next_states)
        self._model.learn(loss.mean() if self._config.per_sample_td_error_enabled else loss)
        self._training_counter += 1
        if self._training_counter % self._config.target_update_frequency == 0:
            self._target_model.soft_update(self._model, self._config.tau)

        return loss.detach().numpy()

    def set_exploration_params(self, epsilon):
        self._config.epsilon = epsilon
