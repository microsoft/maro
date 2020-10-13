# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.utils.trajectory_utils import get_k_step_returns, get_lambda_returns


class ActorCriticHyperParameters:
    """Hyper-parameter set for the Actor-Critic algorithm.

    Args:
        num_actions (int): number of possible actions
        reward_decay (float): reward decay as defined in standard RL terminology
        k (int): number of time steps used in computing returns or return estimates. Defaults to -1, in which case
            rewards are accumulated until the end of the trajectory.
        lmda (float): lambda coefficient used in computing lambda returns. If it is not None, ``k`` will be used as
            the roll-out horizon in computing truncated lambda returns. Otherwise, ``k`` will be used as usual in
            computing returns or multi-step bootstrapped return estimates. Defaults to None.
    """
    __slots__ = ["num_actions", "reward_decay", "k", "lmda"]

    def __init__(self, num_actions: int, reward_decay: float, k: int = -1, lmda: float = None):
        self.num_actions = num_actions
        self.reward_decay = reward_decay
        self.k = k
        self.lmda = lmda


class ActorCritic(AbsAlgorithm):
    """Actor Critic algorithm.

    The Actor-Critic algorithm base on the policy gradient theorem and with REINFORCE and REINFORCE with baseline as
    special cases.

    Args:
        policy_model (nn.Module): model for generating actions given states.
        policy_optimizer_cls: torch optimizer class for the policy model.
        policy_optimizer_params: parameters required for the policy optimizer class.
        hyper_params: hyper-parameter set for the AC algorithm.
        value_model (nn.Module): model for estimating state values. If None, the sequences passed to ``train()`` are
            assumed to end with a terminal state with value zero. If this and the lmda hyper-parameter are both None,
            the returns computed in ``train`` are actual full returns and the algorithm reduces to REINFORCE. Defaults
            to None.
        value_optimizer_cls: torch optimizer class for the value model. Defaults to None.
        value_optimizer_params: parameters required for the value optimizer class. Defaults to None.
        value_loss_func (Callable): loss function for the value model.
    """

    def __init__(self, policy_model: nn.Module, policy_optimizer_cls, policy_optimizer_params,
                 hyper_params: ActorCriticHyperParameters, value_model: nn.Module = None,
                 value_optimizer_cls=None, value_optimizer_params=None, value_loss_func: Callable = None):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._policy_model = policy_model.to(self._device)
        self._policy_optimizer = policy_optimizer_cls(self._policy_model.parameters(), **policy_optimizer_params)
        self._value_model = value_model.to(self._device)
        if self._value_model is not None:
            assert value_optimizer_cls is not None and value_optimizer_params is not None, \
                "value_optimizer_cls and value_optimizer_params should not be None if value model is not None"
            self._value_optimizer = value_optimizer_cls(self._value_model.parameters(), **value_optimizer_params)
        else:
            self._value_optimizer = None
        self._value_loss_func = value_loss_func
        self._hyper_params = hyper_params

    @property
    def model(self):
        return {"policy": self._policy_model, "value": self._value_model}

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        state = torch.from_numpy(state).unsqueeze(0).to(self._device)   # (1, state_dim)
        action_dist = F.softmax(self._policy_model(state), dim=1).squeeze()  # (num_actions,)
        return np.random.choice(self._hyper_params.num_actions, p=action_dist.numpy())

    def train(self, state_sequence: np.ndarray, action_sequence: np.ndarray, reward_sequence: np.ndarray):
        states = torch.from_numpy(state_sequence).to(self._device)   # (N, state_dim)

        if self._value_model is None:
            state_values = state_values_numpy = None
        else:
            state_values = self._value_model(states)
            state_values_numpy = state_values.numpy()

        if self._hyper_params.lmda is None:
            returns = get_k_step_returns(reward_sequence, self._hyper_params.reward_decay,
                                         k=self._hyper_params.k, values=state_values_numpy)
        else:
            returns = get_lambda_returns(reward_sequence, self._hyper_params.reward_decay, self._hyper_params.lmda,
                                         values=state_values_numpy, truncate_steps=self._hyper_params.k)
        # policy model training
        actions = torch.from_numpy(action_sequence).to(self._device)  # (N,)
        action_dist = F.softmax(self._policy_model(states), dim=1)   # (N, num_actions)
        action_prob = action_dist.gather(1, actions.unsqueeze(1))   # (N, 1)
        log_action_prob = torch.log(action_prob).squeeze()   # (N,)
        policy_loss = -(log_action_prob * returns).mean()
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()
        # value model training (if a value model is present)
        if self._value_model is not None:
            value_loss = self._value_loss_func(state_values, returns)
            self._value_optimizer.zero_grad()
            value_loss.backward()
            self._value_optimizer.step()

    def load_trainable_models(self, model_dict):
        self._policy_model = model_dict["policy"]
        self._value_model = model_dict["value"]

    def dump_trainable_models(self):
        return {"policy": self._policy_model, "value": self._value_model}

    def load_trainable_models_from_file(self, path):
        """Load trainable models from disk."""
        model_dict = torch.load(path)
        self._policy_model = model_dict["policy"]
        self._value_model = model_dict["value"]

    def dump_trainable_models_to_file(self, path: str):
        """Dump the algorithm's trainable models to disk."""
        torch.save({"policy": self._policy_model.state_dict(), "value": self._value_model.state_dict()}, path)
