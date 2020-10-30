# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings

import numpy as np
import torch

from .abs_algorithm import AbsAlgorithm
from maro.rl.models.learning_model import LearningModel
from maro.utils import clone


class DDPGHyperParameters:
    """Hyper-parameter set for the DQN algorithm.
    Args:
        num_actions (int): Number of possible actions
        reward_decay (float): Reward decay as defined in standard RL terminology
        policy_target_update_frequency (int): Number of training rounds between policy target model updates.
        value_target_update_frequency (int): Number of training rounds between policy target model updates.
        policy_tau (float): Soft update coefficient for the policy model, e.g.,
            target_model = tau * eval_model + (1-tau) * target_model
        value_tau (float): Soft update coefficient for the value model.
    """
    __slots__ = [
        "num_actions", "reward_decay", "policy_target_update_frequency", "value_target_update_frequency",
        "policy_tau", "value_tau"]

    def __init__(
        self,
        num_actions: int,
        reward_decay: float,
        policy_target_update_frequency: int,
        value_target_update_frequency: int,
        policy_tau: float = 1.0,
        value_tau: float = 1.0
    ):
        self.num_actions = num_actions
        self.reward_decay = reward_decay
        self.policy_target_update_frequency = policy_target_update_frequency
        self.value_target_update_frequency = value_target_update_frequency
        self.policy_tau = policy_tau
        self.value_tau = value_tau


class DDPG(AbsAlgorithm):
    def __init__(
        self,
        policy_model: LearningModel,
        value_model: LearningModel,
        value_loss_func,
        policy_optimizer_cls,
        policy_optimizer_params,
        value_optimizer_cls,
        value_optimizer_params,
        hyper_params: DDPGHyperParameters
    ):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_dict = {
            "policy": policy_model.to(self._device),
            "value": value_model.to(self._device),
            "policy_target": clone(policy_model).to(self._device),
            "value_target": clone(value_model).to(self._device)
        }

        # No gradient computation required for the target models
        for param in self._model_dict["policy_target"].parameters():
            param.requires_grad = False
        for param in self._model_dict["value_target"].parameters():
            param.requires_grad = False

        if policy_optimizer_cls is not None:
            self._policy_optimizer = policy_optimizer_cls(
                self._model_dict["policy"].parameters(), **policy_optimizer_params
            )

        if value_optimizer_cls is not None:
            self._value_optimizer = value_optimizer_cls(
                self._model_dict["value"].parameters(), **value_optimizer_params
            )

        self._value_loss_func = value_loss_func
        self._hyper_params = hyper_params
        self._policy_train_cnt = 0
        self._value_train_cnt = 0

    def choose_action(self, state, epsilon=None):
        state = torch.from_numpy(state).unsqueeze(0).to(self._device)  # (1, state_dim)
        self._model_dict["policy"].eval()
        with torch.no_grad():
            action = self._model_dict["policy"](state)
        return action

    def _train_value_model(
        self, states: torch.tensor, actions: torch.tensor, rewards: torch.tensor, next_states: torch.tensor
    ):
        # value model training
        if hasattr(self, "_value_optimizer"):
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(1)  # (N, 1)
            current_q_values = self._model_dict["value"](torch.cat([states, actions])).squeeze(1)  # (N,)
            next_actions = self._model_dict["policy_target"](states).unsqueeze(dim=1)
            next_q_values = self._model_dict["value_target"](torch.cat([next_states, next_actions])).squeeze(1)  # (N,)
            target_q_values = (rewards + self._hyper_params.reward_decay * next_q_values).detach()  # (N,)
            loss = self._value_loss_func(current_q_values, target_q_values)
            self._model_dict["value"].train()
            self._value_optimizer.zero_grad()
            loss.backward()
            self._value_optimizer.step()
            self._value_train_cnt += 1
            if self._value_train_cnt % self._hyper_params.value_target_update_frequency == 0:
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
            if self._policy_train_cnt % self._hyper_params.policy_target_update_frequency == 0:
                self._update_target_model("policy")

    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
        if not hasattr(self, "_value_optimizer") and not hasattr(self, "_policy_optimizer"):
            warnings.warn(f"No value optimizer or policy optimizer found. Make sure you are using the right "
                          f"AgentManagerMode.")
            return

        states = torch.from_numpy(states).to(self._device)  # (N, state_dim)
        actions = torch.from_numpy(actions).to(self._device).unsqueeze(dim=1)  # (N, 1)
        rewards = torch.from_numpy(rewards).to(self._device)  # (N,)
        next_states = torch.from_numpy(next_states).to(self._device)  # (N, state_dim)

        self._train_value_model(states, actions, rewards, next_states)
        self._train_policy_model(states)

    def _update_target_model(self, which: str):
        if which not in {"policy", "value"}:
            raise ValueError(f"unrecognized member: {which}")
        if hasattr(self, f"_{which}_optimizer"):
            tau = getattr(self._hyper_params, f"{which}_tau")
            for eval_params, target_params in zip(
                self._model_dict[which].parameters(), self._model_dict[f"{which}_target"].parameters()
            ):
                target_params.data = tau * eval_params.data + (1 - tau) * target_params.data

