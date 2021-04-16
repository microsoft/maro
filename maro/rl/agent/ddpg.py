# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np
import torch

from maro.rl.cls_index import TORCH_LOSS_CLS
from maro.rl.exploration import NoiseExplorer
from maro.rl.model import SimpleMultiHeadModel
from maro.rl.storage import SimpleStore
from maro.rl.utils import get_cls
from maro.utils.exception.rl_toolkit_exception import UnrecognizedTask

from .agent import AbsAgent, GenericAgentConfig


class DDPGConfig:
    """Configuration for the DDPG algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        target_update_freq (int): Number of training rounds between policy target model updates.
        q_value_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for
            the Q-value loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        policy_loss_coefficient (float): The coefficient for policy loss in the total loss function, e.g.,
            loss = q_value_loss + ``policy_loss_coefficient`` * policy_loss. Defaults to 1.0.
        soft_update_coefficient (float): Soft update coefficient, e.g.,
            target_model = (soft_update_coefficient) * eval_model + (1-soft_update_coefficient) * target_model.
            Defaults to 1.0.
    """
    __slots__ = [
        "reward_discount", "q_value_loss_func", "target_update_freq", "policy_loss_coefficient",
        "soft_update_coefficient"
    ]

    def __init__(
        self,
        reward_discount: float,
        target_update_freq: int,
        q_value_loss_cls="mse",
        policy_loss_coefficient: float = 1.0,
        soft_update_coefficient: float = 1.0,
    ):
        self.reward_discount = reward_discount
        self.target_update_freq = target_update_freq
        self.q_value_loss_func = get_cls(q_value_loss_cls, TORCH_LOSS_CLS)()
        self.policy_loss_coefficient = policy_loss_coefficient
        self.soft_update_coefficient = soft_update_coefficient


class DDPG(AbsAgent):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        https://arxiv.org/pdf/1509.02971.pdf
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg

    Args:
        model (AbsCoreModel): Task model or container of task models required by the algorithm.
        algorithm_config: Algorithm-specific configuration.
        generic_config (GenericAgentConfig): Non-algorithm-specific configuration.  
        experience_memory (SimpleStore): Experience memory for the agent. If None, an experience memory will be
            created at init time. Defaults to None.
        explorer (NoiseExplorer): An NoiseExplorer instance for generating exploratory actions. Defaults to None.
    """
    def __init__(
        self,
        model: SimpleMultiHeadModel,
        algorithm_config,
        generic_config: GenericAgentConfig,
        experience_memory: SimpleStore = None,
        explorer: NoiseExplorer = None
    ):
        if model.task_names is None or set(model.task_names) != {"policy", "q_value"}:
            raise UnrecognizedTask(f"Expected model task names 'policy' and 'q_value', but got {model.task_names}")
        super().__init__(model, algorithm_config, generic_config, experience_memory=experience_memory)
        self._explorer = explorer
        self._target_model = model.copy() if model.trainable else None
        self._train_cnt = 0

    def choose_action(self, state) -> Union[float, np.ndarray]:
        state = torch.from_numpy(state).to(self.device)
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(dim=0)

        action = self.model(state, task_name="policy", training=False).data.cpu().numpy()
        action_dim = action.shape[1]
        if self._explorer:
            action = self._explorer(action)

        if action_dim == 1:
            action = action.squeeze(axis=1)

        return action[0] if is_single else action

    def step(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
        states = torch.from_numpy(states).to(self.device)
        actual_actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        if len(actual_actions.shape) == 1:
            actual_actions = actual_actions.unsqueeze(dim=1)  # (N, 1)

        current_q_values = self.model(torch.cat([states, actual_actions], dim=1), task_name="q_value")
        current_q_values = current_q_values.squeeze(dim=1)  # (N,)
        next_actions = self._target_model(states, task_name="policy", training=False)
        next_q_values = self._target_model(
            torch.cat([next_states, next_actions], dim=1), task_name="q_value", training=False
        ).squeeze(1)  # (N,)
        target_q_values = (rewards + self.algorithm_config.reward_discount * next_q_values).detach()  # (N,)
        q_value_loss = self.algorithm_config.q_value_loss_func(current_q_values, target_q_values)
        actions_from_model = self.model(states, task_name="policy")
        policy_loss = -self.model(torch.cat([states, actions_from_model], dim=1), task_name="q_value").mean()
        self.model.step(q_value_loss + self.algorithm_config.policy_loss_coefficient * policy_loss)
        self._train_cnt += 1
        if self._train_cnt % self.algorithm_config.target_update_freq == 0:
            self._target_model.soft_update(self.model, self.algorithm_config.soft_update_coefficient)
