# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import torch.nn as nn

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.utils import clone


class DQNHyperParams:
    """Hyper-parameter set for the DQN algorithm.

    Args:
        num_actions (int): number of possible actions
        reward_decay (float): reward decay as defined in standard RL terminology
        num_training_rounds_per_target_replacement (int): number of training frequency of target model replacement
        tau (float): soft update coefficient, e.g., target_model = tau * eval_model + (1-tau) * target_model
    """
    __slots__ = ["num_actions", "reward_decay", "num_training_rounds_per_target_replacement", "tau"]
    def __init__(self, num_actions: int, reward_decay: float, num_training_rounds_per_target_replacement: int,
                 tau: float = 1.0):
        self.num_actions = num_actions
        self.reward_decay = reward_decay
        self.num_training_rounds_per_target_replacement = num_training_rounds_per_target_replacement
        self.tau = tau


class DQN(AbsAlgorithm):
    """The Deep-Q-Networks algorithm.

    The model must contain the key `eval`. Optionally a model corresponding to the key `target` can be
    provided. If the key `target` is absent or model[`target`] is None, the target model will be a deep
    copy of the provided eval model.
    """
    def __init__(self, eval_model: nn.Module, optimizer_cls, optimizer_params, loss_func, hyper_params: DQNHyperParams,
                 target_model: nn.Module = None):
        super().__init__()
        self._eval_model = eval_model
        self._target_model = clone(eval_model) if target_model is None else target_model
        self._optimizer = optimizer_cls(self._eval_model.parameters(), **optimizer_params)
        self._loss_func = loss_func
        self._hyper_params = hyper_params
        self._train_cnt = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def eval_model(self):
        return self._eval_model

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        if epsilon is None or np.random.rand() > epsilon:
            state = torch.from_numpy(state).unsqueeze(0)
            self._eval_model.eval()
            with torch.no_grad():
                q_values = self._eval_model(state)
            return q_values.argmax(dim=1).item()

        return np.random.choice(self._hyper_params.num_actions)

    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
        states = torch.from_numpy(states).to(self._device)  # (N, state_dim)
        actions = torch.from_numpy(actions).to(self._device)  # (N,)
        rewards = torch.from_numpy(rewards).to(self._device)   # (N,)
        next_states = torch.from_numpy(next_states).to(self._device)  # (N, state_dim)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)   # (N, 1)
        current_q_values = self._eval_model(states).gather(1, actions).squeeze(1)   # (N,)
        next_q_values = self._target_model(next_states).max(dim=1)[0]   # (N,)
        target_q_values = (rewards + self._hyper_params.reward_decay * next_q_values).detach()   # (N,)
        loss = self._loss_func(current_q_values, target_q_values)
        self._eval_model.train()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._train_cnt += 1
        if self._train_cnt % self._hyper_params.num_training_rounds_per_target_replacement == 0:
            self._update_target_model()

        return np.abs((current_q_values - target_q_values).detach().numpy())

    def _update_target_model(self):
        for eval_params, target_params in zip(self._eval_model.parameters(), self._target_model.parameters()):
            target_params.data = (
                self._hyper_params.tau * eval_params.data + (1 - self._hyper_params.tau) * target_params.data
            )

    def load_trainable_models(self, eval_model):
        """Load the eval model from memory."""
        self._eval_model = eval_model

    def dump_trainable_models(self):
        """Return the eval model."""
        return self._eval_model

    def load_trainable_models_from_file(self, path):
        """Load the eval model from disk."""
        self._eval_model = torch.load(path)

    def dump_trainable_models_to_file(self, path: str):
        """Dump the eval model to disk."""
        torch.save(self._eval_model.state_dict(), path)
