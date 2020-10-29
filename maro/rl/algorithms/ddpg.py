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
        policy_target_update_period (int): Number of training rounds between policy target model updates.
        value_target_update_period (int): Number of training rounds between policy target model updates.
        policy_tau (float): Soft update coefficient for the policy model, e.g.,
            target_model = tau * eval_model + (1-tau) * target_model
        value_tau (float): Soft update coefficient for the value model.
    """
    __slots__ = [
        "num_actions", "reward_decay", "policy_target_update_period", "value_target_update_period",
        "policy_tau", "value_tau"]

    def __init__(
        self,
        num_actions: int,
        reward_decay: float,
        policy_target_update_period: int,
        value_target_update_period: int,
        policy_tau: float = 1.0,
        value_tau: float = 1.0
    ):
        self.num_actions = num_actions
        self.reward_decay = reward_decay
        self.policy_target_update_period = policy_target_update_period
        self.value_target_update_period = value_target_update_period
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
        self._training_cnt = 0

    def choose_action(self, state, epsilon=None):
        state = torch.from_numpy(state).unsqueeze(0).to(self._device)  # (1, state_dim)
        self._model_dict["policy"].eval()
        with torch.no_grad():
            action = self._model_dict["policy"](state)
        return action

    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
        if not hasattr(self, "_value_optimizer") and not hasattr(self, "_policy_optimizer"):
            warnings.warn(f"No value optimizer or policy optimizer found. Make sure you are using the right "
                          f"AgentManagerMode.")
            return

        states = torch.from_numpy(states).to(self._device)  # (N, state_dim)
        actions = torch.from_numpy(actions).to(self._device)  # (N,)

        # value model training
        if hasattr(self, "_value_optimizer"):
            rewards = torch.from_numpy(rewards).to(self._device)  # (N,)
            next_states = torch.from_numpy(next_states).to(self._device)  # (N, state_dim)
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(1)  # (N, 1)
            current_q_values = self._model_dict["value"](states).gather(1, actions).squeeze(1)  # (N,)
            next_q_values = self._model_dict["value"](next_states).max(dim=1)[0]  # (N,)
            target_q_values = (rewards + self._hyper_params.reward_decay * next_q_values).detach()  # (N,)
            loss = self._loss_func(current_q_values, target_q_values)
            self._model_dict["eval"].train()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._train_cnt += 1
            if self._train_cnt % self._hyper_params.value_target_update_period == 0:
                self._update_target_model("value")

        # # policy model training
        # if hasattr(self, "_policy_optimizer"):
            
    def _update_target_model(self, which: str):
        if which not in {"policy", "value"}:
            raise ValueError(f"unrecognized member: {which}")
        if hasattr(self, f"_{which}_optimizer"):
            tau = getattr(self._hyper_params, f"{which}_tau")
            for eval_params, target_params in zip(
                self._model_dict[which].parameters(), self._model_dict[f"{which}_target"].parameters()
            ):
                target_params.data = tau * eval_params.data + (1 - tau) * target_params.data

