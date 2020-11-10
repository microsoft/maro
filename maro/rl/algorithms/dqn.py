# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.models.abs_learning_model import AbsLearningModel
from maro.rl.models.learning_model import MultiTaskLearningModel


class DuelingHead(Enum):
    STATE_VALUE = "state_value"
    ADVANTAGE = "advantage"


class DQNConfig:
    """Configuration for the DQN algorithm.

    Args:
        num_actions (int): Number of possible actions.
        reward_decay (float): Reward decay as defined in standard RL terminology.
        target_update_frequency (int): Number of training rounds between target model updates.
        tau (float): Soft update coefficient, i.e., target_model = tau * eval_model + (1 - tau) * target_model.
        is_double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
        advantage_mode (str): advantage mode for the dueling architecture. Defaults to None, in which
            case it is assumed that the regular Q-value model is used.
        per_sample_td_error_enabled (bool): If True, per-sample TD errors will be returned by the DQN's train()
            method. Defaults to False.
    """
    __slots__ = [
        "num_actions", "reward_decay", "target_update_frequency", "tau", "is_double", "advantage_mode",
        "per_sample_td_error_enabled"
    ]

    def __init__(
        self,
        num_actions: int,
        reward_decay: float,
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


class DQN(AbsAlgorithm):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        core_model (nn.Module): Q-value model.
        loss_cls (Callable): Loss function class for computing TD error.
        config: Configuration for DQN algorithm.
    """
    def __init__(self, core_model: AbsLearningModel, loss_cls: Callable, config: DQNConfig):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._config = config
        self._loss_func = loss_cls(reduction="none" if self._config.per_sample_td_error_enabled else "mean")
        self._training_counter = 0

        if self._config.advantage_mode is not None:
            assert isinstance(core_model, MultiTaskLearningModel), \
                "core_model must be a MultiTaskLearningModel if dueling architecture is used."
            assert DuelingHead.STATE_VALUE.value in core_model.heads, \
                "core_mode; must have a task head named 'state_value'"
            assert DuelingHead.ADVANTAGE.value in core_model.heads, \
                "core_model must have a task head named 'advantage'"

        self._eval_model = core_model.to(self._device)
        self._target_model = core_model.copy().to(self._device) if self._eval_model.is_trainable else None

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        if epsilon is None or np.random.rand() > epsilon:
            state = torch.from_numpy(state).unsqueeze(0)
            self._eval_model.eval()
            with torch.no_grad():
                q_values = self._get_q_values(state)
            return q_values.argmax(dim=1).item()

        return np.random.choice(self._config.num_actions)

    def _compute_td_errors(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray
    ):
        states = torch.from_numpy(states).to(self._device)  # (N, state_dim)
        actions = torch.from_numpy(actions).to(self._device)  # (N,)
        rewards = torch.from_numpy(rewards).to(self._device)  # (N,)
        next_states = torch.from_numpy(next_states).to(self._device)  # (N, state_dim)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)  # (N, 1)
        current_q_values_for_all_actions = self._get_q_values(states)
        current_q_values = current_q_values_for_all_actions.gather(1, actions).squeeze(1)  # (N,)
        next_q_values = self._get_next_q_values(current_q_values_for_all_actions, next_states)  # (N,)
        target_q_values = (rewards + self._config.reward_decay * next_q_values).detach()  # (N,)
        return self._loss_func(current_q_values, target_q_values)

    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
        loss = self._compute_td_errors(states, actions, rewards, next_states)
        self._eval_model.step(loss.mean() if self._config.per_sample_td_error_enabled else loss)
        self._training_counter += 1
        if self._training_counter % self._config.target_update_frequency == 0:
            self._update_targets()

        return loss.detach().numpy()

    def _update_targets(self):
        for eval_params, target_params in zip(
            self._eval_model.parameters(), self._target_model.parameters()
        ):
            target_params.data = (
                self._config.tau * eval_params.data + (1 - self._config.tau) * target_params.data
            )

    def _get_q_values(self, states, is_target: bool = False):
        if self._config.advantage_mode is not None:
            output = self._target_model(states) if is_target else self._eval_model(states)
            state_values = output["state_value"]
            advantages = output["advantage"]
            # Use mean or max correction to address the identifiability issue
            corrections = advantages.mean(1) if self._config.advantage_mode == "mean" else advantages.max(1)[0]
            q_values = state_values + advantages - corrections.unsqueeze(1)
            return q_values
        else:
            model = self._target_model if is_target else self._eval_model
            return model(states)

    def _get_next_q_values(self, current_q_values_all, next_states):
        if self._config.is_double:
            actions = current_q_values_all.max(dim=1)[1].unsqueeze(1)
            return self._get_q_values(next_states, is_target=True).gather(1, actions).squeeze(1)  # (N,)
        else:
            return self._get_q_values(next_states, is_target=True).max(dim=1)[0]   # (N,)

    def load_models(self, eval_model):
        """Load the evaluation model from memory."""
        self._eval_model.load_state_dict(eval_model)

    def dump_models(self):
        """Return the evaluation model."""
        return self._eval_model.state_dict()

    def load_models_from_file(self, path):
        """Load the evaluation model from disk."""
        self._eval_model.load_state_dict(torch.load(path))

    def dump_models_to_file(self, path: str):
        """Dump the evaluation model to disk."""
        torch.save(self._eval_model.state_dict(), path)
