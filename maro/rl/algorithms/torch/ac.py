# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.utils.trajectory_utils import get_lambda_returns


class ActorCriticHyperParameters:
    """Hyper-parameter set for the Actor-Critic algorithm.

    Args:
        num_actions (int): number of possible actions
        reward_decay (float): reward decay as defined in standard RL terminology
        k (int): number of time steps used in computing returns or return estimates. Defaults to -1, in which case
            rewards are accumulated until the end of the trajectory.
        lamb (float): lambda coefficient used in computing lambda returns. Defaults to 1.0, in which case the usual
            k-step return is computed.
    """
    __slots__ = ["num_actions", "reward_decay", "k", "lamb"]

    def __init__(self, num_actions: int, reward_decay: float, k: int = -1, lamb: float = 1.0):
        self.num_actions = num_actions
        self.reward_decay = reward_decay
        self.k = k
        self.lamb = lamb


class ActorCritic(AbsAlgorithm):
    """Actor Critic algorithm with separate policy and value models (no shared layers).

    The Actor-Critic algorithm base on the policy gradient theorem.

    Args:
        policy_model (nn.Module): model for generating actions given states.
        value_model (nn.Module): model for estimating state values.
        value_loss_func (Callable): loss function for the value model.
        policy_optimizer_cls: torch optimizer class for the policy model.
        policy_optimizer_params: parameters required for the policy optimizer class.
        value_optimizer_cls: torch optimizer class for the value model.
        value_optimizer_params: parameters required for the value optimizer class.
        hyper_params: hyper-parameter set for the AC algorithm.
    """

    def __init__(self,
                 policy_model: nn.Module,
                 value_model: nn.Module,
                 value_loss_func: Callable,
                 policy_optimizer_cls,
                 policy_optimizer_params,
                 value_optimizer_cls,
                 value_optimizer_params,
                 hyper_params: ActorCriticHyperParameters):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._policy_model = policy_model.to(self._device)
        self._value_model = value_model.to(self._device)
        self._policy_optimizer = policy_optimizer_cls(self._policy_model.parameters(), **policy_optimizer_params)
        self._value_optimizer = value_optimizer_cls(self._value_model.parameters(), **value_optimizer_params)
        self._value_loss_func = value_loss_func
        self._hyper_params = hyper_params

    @property
    def model(self):
        return {"policy": self._policy_model, "value": self._value_model}

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        state = torch.from_numpy(state).unsqueeze(0).to(self._device)   # (1, state_dim)
        action_dist = self._policy_model(state).squeeze()  # (num_actions,)
        return np.random.choice(self._hyper_params.num_actions, p=action_dist.numpy())

    def train(self, state_sequence: np.ndarray, action_sequence: np.ndarray, reward_sequence: np.ndarray):
        states = torch.from_numpy(state_sequence).to(self._device)   # (N, state_dim)
        state_values = self._value_model(states)
        state_values_numpy = state_values.numpy()
        returns = get_lambda_returns(reward_sequence, self._hyper_params.reward_decay, self._hyper_params.lamb,
                                     k=self._hyper_params.k, values=state_values_numpy)
        advantages = returns - state_values
        # policy model training
        actions = torch.from_numpy(action_sequence).to(self._device)  # (N,)
        action_prob = self._policy_model(states).gather(1, actions.unsqueeze(1)).squeeze()  # (N,)
        policy_loss = -(torch.log(action_prob) * advantages).mean()
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # value model training
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


class ActorCriticWithSharedLayers(AbsAlgorithm):
    """Actor Critic algorithm where policy and value models have shared layers.

    Args:
        policy_value_model (nn.Module): model for generating action distributions and values for given states using
            shared bottom layers. The model, when called, must return (value, action distribution).
        value_loss_func (Callable): loss function for the value model.
        optimizer_cls: torch optimizer class for the policy model.
        optimizer_params: parameters required for the policy optimizer class.
        hyper_params: hyper-parameter set for the AC algorithm.
    """

    def __init__(self,
                 policy_value_model: nn.Module,
                 value_loss_func: Callable,
                 optimizer_cls,
                 optimizer_params,
                 hyper_params: ActorCriticHyperParameters):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._policy_value_model = policy_value_model.to(self._device)
        self._optimizer = optimizer_cls(self._policy_value_model.parameters(), **optimizer_params)
        self._value_loss_func = value_loss_func
        self._hyper_params = hyper_params

    @property
    def model(self):
        return self._policy_value_model

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        state = torch.from_numpy(state).unsqueeze(0).to(self._device)   # (1, state_dim)
        action_dist = self._policy_value_model(state)[1].squeeze()  # (num_actions,)
        return np.random.choice(self._hyper_params.num_actions, p=action_dist.numpy())

    def train(self, state_sequence: np.ndarray, action_sequence: np.ndarray, reward_sequence: np.ndarray):
        states = torch.from_numpy(state_sequence).to(self._device)   # (N, state_dim)
        state_values, action_distribution = self._policy_value_model(states)
        state_values_numpy = state_values.numpy()
        returns = get_lambda_returns(reward_sequence, self._hyper_params.reward_decay, self._hyper_params.lamb,
                                     k=self._hyper_params.k, values=state_values_numpy)
        advantages = returns - state_values
        actions = torch.from_numpy(action_sequence).to(self._device)  # (N,)
        action_prob = action_distribution.gather(1, actions.unsqueeze(1)).squeeze()   # (N,)
        policy_loss = -(torch.log(action_prob) * advantages).mean()
        value_loss = self._value_loss_func(state_values, returns)
        loss = policy_loss + value_loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def load_trainable_models(self, policy_value_model):
        self._policy_value_model = policy_value_model

    def dump_trainable_models(self):
        return self._policy_value_model

    def load_trainable_models_from_file(self, path):
        """Load trainable models from disk."""
        self._policy_value_model = torch.load(path)

    def dump_trainable_models_to_file(self, path: str):
        """Dump the algorithm's trainable models to disk."""
        torch.save(self._policy_value_model.state_dict(), path)
