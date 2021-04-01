# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np
import torch

from maro.rl.model import SimpleMultiHeadModel
from maro.rl.utils import get_max, get_td_errors, select_by_actions
from maro.utils.exception.rl_toolkit_exception import UnrecognizedTask

from .abs_agent import AbsAgent


class DQNConfig:
    """Configuration for the DQN algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        epsilon (float): Exploration rate for epsilon-greedy exploration. Defaults to None.
        tau (float): Soft update coefficient, i.e., target_model = tau * eval_model + (1 - tau) * target_model.
        double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
        advantage_type (str): Advantage mode for the dueling architecture. Defaults to None, in which
            case it is assumed that the regular Q-value model is used.
        loss_cls: Loss function class for evaluating TD errors. Defaults to torch.nn.MSELoss.
        target_update_freq (int): Number of training rounds between target model updates.
    """
    __slots__ = [
        "reward_discount", "target_update_freq", "epsilon", "tau", "double", "advantage_type", "loss_func"
    ]

    def __init__(
        self,
        reward_discount: float,
        target_update_freq: int,
        epsilon: float = .0,
        tau: float = 0.1,
        double: bool = True,
        advantage_type: str = None,
        loss_cls=torch.nn.MSELoss
    ):
        self.reward_discount = reward_discount
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon
        self.tau = tau
        self.double = double
        self.advantage_type = advantage_type
        self.loss_func = loss_cls(reduction="none")


class DQN(AbsAgent):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        model (SimpleMultiHeadModel): Q-value model.
        config: Configuration for DQN algorithm.
    """
    def __init__(self, model: SimpleMultiHeadModel, config: DQNConfig):
        if (config.advantage_type is not None and
                (model.task_names is None or set(model.task_names) != {"state_value", "advantage"})):
            raise UnrecognizedTask(
                f"Expected model task names 'state_value' and 'advantage' since dueling DQN is used, "
                f"got {model.task_names}"
            )
        super().__init__(model, config)
        self._training_counter = 0
        self._target_model = model.copy() if model.trainable else None

    def choose_action(self, state: np.ndarray) -> Union[int, np.ndarray]:
        state = torch.from_numpy(state)
        if self.device:
            state = state.to(self.device)
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(dim=0)

        q_values = self._get_q_values(state, training=False)
        num_actions = q_values.shape[1]
        greedy_action = q_values.argmax(dim=1).data.cpu()
        # No exploration
        if self.config.epsilon == .0:
            return greedy_action.item() if is_single else greedy_action.numpy()

        if is_single:
            return greedy_action if np.random.random() > self.config.epsilon else np.random.choice(num_actions)

        # batch inference
        return np.array([
            act if np.random.random() > self.config.epsilon else np.random.choice(num_actions)
            for act in greedy_action
        ])

    def learn(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)

        if self.device:
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)

        q_all = self._get_q_values(states)
        q = select_by_actions(q_all, actions)
        next_q_all_target = self._get_q_values(next_states, is_eval=False, training=False)
        if self.config.double:
            next_q_all_eval = self._get_q_values(next_states, training=False)
            next_q = select_by_actions(next_q_all_target, next_q_all_eval.max(dim=1)[1])  # (N,)
        else:
            next_q, _ = get_max(next_q_all_target)  # (N,)

        loss = get_td_errors(q, next_q, rewards, self.config.reward_discount, loss_func=self.config.loss_func)
        self.model.step(loss.mean())
        self._training_counter += 1
        if self._training_counter % self.config.target_update_freq == 0:
            self._target_model.soft_update(self.model, self.config.tau)

        return loss.detach().numpy()

    def set_exploration_params(self, epsilon):
        self.config.epsilon = epsilon

    def _get_q_values(self, states: torch.Tensor, is_eval: bool = True, training: bool = True):
        output = self.model(states, training=training) if is_eval else self._target_model(states, training=False)
        if self.config.advantage_type is None:
            return output
        else:
            state_values = output["state_value"]
            advantages = output["advantage"]
            # Use mean or max correction to address the identifiability issue
            corrections = advantages.mean(1) if self.config.advantage_type == "mean" else advantages.max(1)[0]
            return state_values + advantages - corrections.unsqueeze(1)
