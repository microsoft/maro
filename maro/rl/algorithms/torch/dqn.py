# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from copy import deepcopy

import numpy as np
import torch

from maro.rl.algorithms.torch.algorithm import Algorithm
from maro.rl.common import ExperienceKey, ExperienceInfoKey


class DQNHyperParams:
    def __init__(self, num_actions: int, replace_target_frequency: int, tau: float):
        self.num_actions = num_actions
        self.replace_target_frequency = replace_target_frequency
        self.tau = tau


class DQN(Algorithm):
    def __init__(self, model_dict, optimizer_opt, loss_func_dict, hyper_params: DQNHyperParams):
        if model_dict.get("target", None) is None:
            model_dict['target'] = deepcopy(model_dict['eval'])
        super().__init__(model_dict, optimizer_opt, loss_func_dict, hyper_params)
        self._train_cnt = 0

    def choose_action(self, state, epsilon: float = None):
        if epsilon is None or random.random() > epsilon:
            state = torch.from_numpy(state).unsqueeze(0)
            self._model_dict["eval"].eval()
            with torch.no_grad():
                q_values = self._model_dict["eval"](state)
            best_action_idx = q_values.argmax(dim=1).item()
            return best_action_idx

        return random.choice(range(self._hyper_params.num_actions))

    def train_on_batch(self, batch):
        state = torch.from_numpy(batch[ExperienceKey.STATE])
        action = torch.from_numpy(batch[ExperienceKey.ACTION])
        reward = torch.from_numpy(batch[ExperienceKey.REWARD]).squeeze(1)
        next_state = torch.from_numpy(batch[ExperienceKey.NEXT_STATE])
        discount = torch.from_numpy(batch[ExperienceInfoKey.DISCOUNT])
        q_value = self._model_dict["eval"](state).gather(1, action).squeeze(1)
        target = (reward + discount * self._model_dict["target"](next_state).max(dim=1)[0]).detach()
        loss = self._loss_func_dict["eval"](q_value, target)
        self._model_dict["eval"].train()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._train_cnt += 1
        if self._train_cnt % self._hyper_params.replace_target_frequency == 0:
            self._update_target_model()

        return np.abs((q_value - target).detach().numpy())

    def _update_target_model(self):
        for evl, target in zip(self._model_dict["eval"].parameters(), self._model_dict["target"].parameters()):
            target.data = self._hyper_params.tau * evl.data + (1 - self._hyper_params.tau) * target.data
