# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.utils import clone


class DQNHyperParams:
    """Hyper-parameter set for the DQN algorithm.

    Args:
        num_actions (int): Number of possible actions.
        reward_decay (float): Reward decay as defined in standard RL terminology.
        target_update_frequency (int): Number of training rounds between target model updates.
        tau (float): Soft update coefficient, i.e., target_model = tau * eval_model + (1 - tau) * target_model.
        is_double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
    """
    __slots__ = ["num_actions", "reward_decay", "target_update_frequency", "tau", "is_double"]

    def __init__(
        self,
        num_actions: int,
        reward_decay: float,
        target_update_frequency: int,
        tau: float = 0.1,
        is_double: bool = True
    ):
        self.num_actions = num_actions
        self.reward_decay = reward_decay
        self.target_update_frequency = target_update_frequency
        self.tau = tau
        self.is_double = is_double


class DQN(AbsAlgorithm):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        q_model (nn.Module): Q-value model.
        optimizer_cls (torch.optim.Optimizer): Torch optimizer class for the value model. If this is None, the eval
            model is not trainable.
        optimizer_params (dict): Parameters required for the optimizer class for the value model.
        loss_func (Callable): Loss function for the value model.
        hyper_params: Hyper-parameter set for the DQN algorithm.
    """
    def __init__(
        self,
        q_model: nn.Module,
        optimizer_cls: torch.optim.Optimizer,
        optimizer_params: dict,
        loss_func: Callable,
        hyper_params: DQNHyperParams,
    ):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._hyper_params = hyper_params
        self._is_trainable = optimizer_cls is not None
        self._loss_func = loss_func
        self._training_counter = 0

        self._eval_model = q_model.to(self._device)
        self._target_model = clone(q_model).to(self._device) if self._is_trainable else None

        if self._target_model is not None:
            # No gradient computation required for the target model
            for param in self._target_model.parameters():
                param.requires_grad = False

        self._optimizer = optimizer_cls(self._eval_model.parameters(), **optimizer_params)

    @property
    def is_trainable(self):
        return self._is_trainable

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        if epsilon is None or np.random.rand() > epsilon:
            state = torch.from_numpy(state).unsqueeze(0)
            self._eval_model.eval()
            with torch.no_grad():
                q_values = self._get_q_values(state)
            return q_values.argmax(dim=1).item()

        return np.random.choice(self._hyper_params.num_actions)

    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
        if not self._is_trainable:
            return

        states = torch.from_numpy(states).to(self._device)  # (N, state_dim)
        actions = torch.from_numpy(actions).to(self._device)  # (N,)
        rewards = torch.from_numpy(rewards).to(self._device)   # (N,)
        next_states = torch.from_numpy(next_states).to(self._device)  # (N, state_dim)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)   # (N, 1)
        current_q_values_for_all_actions = self._get_q_values(states)
        current_q_values = current_q_values_for_all_actions.gather(1, actions).squeeze(1)   # (N,)
        next_q_values = self._get_next_q_values(current_q_values_for_all_actions, next_states)   # (N,)
        target_q_values = (rewards + self._hyper_params.reward_decay * next_q_values).detach()   # (N,)
        loss = self._loss_func(current_q_values, target_q_values)
        self._eval_model.train()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._training_counter += 1
        if self._training_counter % self._hyper_params.target_update_frequency == 0:
            self._update_targets()

        return np.abs((current_q_values - target_q_values).detach().numpy())

    def _update_targets(self):
        for eval_params, target_params in zip(
            self._eval_model.parameters(), self._target_model.parameters()
        ):
            target_params.data = (
                self._hyper_params.tau * eval_params.data + (1 - self._hyper_params.tau) * target_params.data
            )

    def _get_q_values(self, states, is_target: bool = False):
        model = self._target_model if is_target else self._eval_model
        return model(states)

    def _get_next_q_values(self, current_q_values_all, next_states):
        if self._hyper_params.is_double:
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
