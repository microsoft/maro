# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
from typing import Union

import numpy as np
import torch

from maro.rl.model import SimpleMultiHeadModel
from maro.rl.storage import SimpleStore
from maro.utils.exception.rl_toolkit_exception import UnrecognizedTask

from .abs_agent import AbsAgent


class DQNConfig:
    """Configuration for the DQN algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        num_batches (int): Number of batches to train the DQN model on per call to ``train``.
        batch_size (int): Mini-batch size.
        epsilon (float): Exploration rate for epsilon-greedy exploration. Defaults to None.
        min_exp_to_train (int): Minimum number of experiences required for training. Defaults to 0.
        tau (float): Soft update coefficient, i.e., target_model = tau * eval_model + (1 - tau) * target_model.
        double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
        advantage_mode (str): Advantage mode for the dueling architecture. Defaults to None, in which
            case it is assumed that the regular Q-value model is used.
        loss_cls: Loss function class for evaluating TD errors. Defaults to torch.nn.MSELoss.
        per_sample_td_error (bool): If True, per-sample TD errors will be returned by the DQN's train()
            method. Defaults to False.
        target_update_freq (int): Number of training rounds between target model updates.
    """
    __slots__ = [
        "reward_discount", "min_exp_to_train", "num_batches", "batch_size", "target_update_freq",
        "epsilon", "tau", "double", "advantage_mode", "per_sample_td_error", "loss_func"
    ]

    def __init__(
        self,
        reward_discount: float,
        num_batches: int,
        batch_size: int,
        target_update_freq: int,
        min_exp_to_train: int = 0,
        epsilon: float = .0,
        tau: float = 0.1,
        double: bool = True,
        advantage_mode: str = None,
        loss_cls=torch.nn.MSELoss,
        per_sample_td_error: bool = False
    ):
        self.reward_discount = reward_discount
        self.min_exp_to_train = min_exp_to_train
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon
        self.tau = tau
        self.double = double
        self.advantage_mode = advantage_mode
        self.per_sample_td_error = per_sample_td_error
        self.loss_func = loss_cls(reduction="none" if per_sample_td_error else "mean")


class DQN(AbsAgent):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        model (SimpleMultiHeadModel): Q-value model.
        config: Configuration for DQN algorithm.
    """
    def __init__(self, model: SimpleMultiHeadModel, config: DQNConfig):
        if (config.advantage_mode is not None and
                (model.task_names is None or set(model.task_names) != {"state_value", "advantage"})):
            raise UnrecognizedTask(
                f"Expected model task names 'state_value' and 'advantage' since dueling DQN is used, "
                f"got {model.task_names}"
            )
        super().__init__(
            model, config,
            experience_pool=SimpleStore(["state", "action", "reward", "next_state", "loss"])
        )
        self._training_counter = 0
        self._target_model = model.copy() if model.is_trainable else None

    def choose_action(self, state: np.ndarray) -> Union[int, np.ndarray]:
        state = torch.from_numpy(state).to(self._device)
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(dim=0)

        q_values = self._get_q_values(self._model, state, training=False)
        num_actions = q_values.shape[1]
        greedy_action = q_values.argmax(dim=1).data
        # No exploration
        if self._config.epsilon == .0:
            return greedy_action.item() if is_single else greedy_action.numpy()

        if is_single:
            return greedy_action if np.random.random() > self._config.epsilon else np.random.choice(num_actions)

        # batch inference
        return np.array([
            act if np.random.random() > self._config.epsilon else np.random.choice(num_actions)
            for act in greedy_action
        ])

    def _get_q_values(self, model, states, training: bool = True):
        if self._config.advantage_mode is not None:
            output = model(states, training=training)
            state_values = output["state_value"]
            advantages = output["advantage"]
            # Use mean or max correction to address the identifiability issue
            corrections = advantages.mean(1) if self._config.advantage_mode == "mean" else advantages.max(1)[0]
            q_values = state_values + advantages - corrections.unsqueeze(1)
            return q_values
        else:
            return model(states, training=training)

    def _get_next_q_values(self, current_q_values_for_all_actions, next_states):
        next_q_values_for_all_actions = self._get_q_values(self._target_model, next_states, training=False)
        if self._config.double:
            actions = current_q_values_for_all_actions.max(dim=1)[1].unsqueeze(1)
            return next_q_values_for_all_actions.gather(1, actions).squeeze(1)  # (N,)
        else:
            return next_q_values_for_all_actions.max(dim=1)[0]   # (N,)

    def _compute_td_errors(self, states, actions, rewards, next_states):
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)  # (N, 1)
        current_q_values_for_all_actions = self._get_q_values(self._model, states)
        current_q_values = current_q_values_for_all_actions.gather(1, actions).squeeze(1)  # (N,)
        next_q_values = self._get_next_q_values(current_q_values_for_all_actions, next_states)  # (N,)
        target_q_values = (rewards + self._config.reward_discount * next_q_values).detach()  # (N,)
        return self._config.loss_func(current_q_values, target_q_values)

    def train(self):
        if len(self._experience_pool) <= self._config.min_exp_to_train:
            return

        for _ in range(self._config.num_batches):
            indexes, sample = self._experience_pool.sample_by_key("loss", self._config.batch_size)
            state = np.asarray(sample["state"])
            action = np.asarray(sample["action"])
            reward = np.asarray(sample["reward"])
            next_state = np.asarray(sample["next_state"])
            loss = self._train_on_batch(state, action, reward, next_state)
            self._experience_pool.update(indexes, {"loss": list(loss)})

    def set_exploration_params(self, epsilon):
        self._config.epsilon = epsilon

    def store_experiences(self, experiences):
        """Store new experiences in the experience pool."""
        self._experience_pool.put(experiences)

    def dump_experience_pool(self, dir_path: str):
        """Dump the experience pool to disk."""
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, self._name), "wb") as fp:
            pickle.dump(self._experience_pool, fp)

    def _train_on_batch(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray):
        states = torch.from_numpy(states).to(self._device)
        actions = torch.from_numpy(actions).to(self._device)
        rewards = torch.from_numpy(rewards).to(self._device)
        next_states = torch.from_numpy(next_states).to(self._device)
        loss = self._compute_td_errors(states, actions, rewards, next_states)
        self._model.learn(loss.mean() if self._config.per_sample_td_error else loss)
        self._training_counter += 1
        if self._training_counter % self._config.target_update_freq == 0:
            self._target_model.soft_update(self._model, self._config.tau)

        return loss.detach().numpy()
