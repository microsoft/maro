# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np
import torch

from maro.rl.algorithms.torch.abs_algorithm import AbsAlgorithm
from maro.utils import clone


class DQNHyperParams:
    """DQN hyper-parameters.

    Args:
        num_actions (int): number of possible actions
        reward_decay (float): reward decay as defined in standard RL terminology
        num_training_rounds_per_target_replacement (int): number of training frequency of target model replacement
        tau (float): soft update coefficient, e.g., target_model = tau * eval_model + (1-tau) * target_model
    """
    __slots__ = ["num_actions", "reward_decay", "num_training_rounds_per_target_replacement", "tau"]

    def __init__(
        self, num_actions: int, reward_decay: float, num_training_rounds_per_target_replacement: int, tau: float = 1.0
    ):
        self.num_actions = num_actions
        self.reward_decay = reward_decay
        self.num_training_rounds_per_target_replacement = num_training_rounds_per_target_replacement
        self.tau = tau


class DQN(AbsAlgorithm):
    """The Deep-Q-Networks algorithm.

    The model_dict must contain the key `eval`. Optionally a model corresponding to the key `target` can be
    provided. If the key `target` is absent or model_dict[`target`] is None, the target model will be a deep
    copy of the provided eval model.
    """
    def __init__(
        self, model_dict: dict, optimizer_opt: Union[dict, tuple], loss_func_dict: dict, hyper_params: DQNHyperParams
    ):
        if model_dict.get("target", None) is None:
            model_dict["target"] = clone(model_dict["eval"])
        super().__init__(model_dict, optimizer_opt, loss_func_dict, hyper_params)
        self._train_cnt = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        if epsilon is None or np.random.rand() > epsilon:
            state = torch.from_numpy(state).unsqueeze(0)
            self._model_dict["eval"].eval()
            with torch.no_grad():
                q_values = self._model_dict["eval"](state)
            best_action_idx = q_values.argmax(dim=1).item()
            return best_action_idx

        return np.random.choice(self._hyper_params.num_actions)

    def _prepare_batch(self, raw_batch):
        return {key: torch.from_numpy(np.asarray(lst)).to(self._device) for key, lst in raw_batch.items()}

    def train(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray):
        state = torch.from_numpy(state).to(self._device)
        action = torch.from_numpy(action).to(self._device)
        reward = torch.from_numpy(reward).to(self._device)
        next_state = torch.from_numpy(next_state).to(self._device)
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        current_q_values = self._model_dict["eval"](state).gather(1, action).squeeze(1)
        next_q_values = self._model_dict["target"](next_state).max(dim=1)[0]
        target_q_values = (reward + self._hyper_params.reward_decay * next_q_values).detach()
        loss = self._loss_func_dict["eval"](current_q_values, target_q_values)
        self._model_dict["eval"].train()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._train_cnt += 1
        if self._train_cnt % self._hyper_params.num_training_rounds_per_target_replacement == 0:
            self._update_target_model()

        return np.abs((current_q_values - target_q_values).detach().numpy())

    def _update_target_model(self):
        for eval_params, target_params in zip(
            self._model_dict["eval"].parameters(), self._model_dict["target"].parameters()
        ):
            target_params.data = (
                self._hyper_params.tau * eval_params.data + (1 - self._hyper_params.tau) * target_params.data
            )
