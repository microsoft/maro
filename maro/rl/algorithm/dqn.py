# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np
import torch

from maro.rl.model import QEstimatorForDiscreteActions
from maro.rl.utils import get_max, get_sampler_cls, get_td_errors, get_torch_loss_cls, select_by_actions

from .abs_agent import AbsAlgorithm


class DQNConfig:
    """Configuration for the DQN algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        target_update_freq (int): Number of training rounds between target model updates.
        train_iters (int): Number of batches to train the model on in each call to ``learn``.
        batch_size (int): Experience minibatch size.
        sampler_cls: A string indicating the sampler class or a custom sampler class that provides the ``sample``
            interface. Defaults to "uniform".
        sampler_params (dict): Parameters for the sampler class. Defaults to None.
        epsilon (float): Exploration rate for epsilon-greedy exploration. Defaults to None.
        soft_update_coefficient (float): Soft update coefficient, e.g.,
            target_model = (soft_update_coefficient) * eval_model + (1-soft_update_coefficient) * target_model.
            Defaults to 1.0.
        double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
        advantage_type (str): Advantage mode for the dueling architecture. Defaults to None, in which
            case it is assumed that the regular Q-value model is used.
        loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class. If it is a string,
            it must be a key in ``TORCH_LOSS``. Defaults to "mse".
    """
    __slots__ = [
        "reward_discount", "target_update_freq", "train_iters", "batch_size", "sampler_cls", "sampler_params",
        "epsilon", "soft_update_coefficient", "double", "advantage_type", "loss_func"
    ]

    def __init__(
        self,
        reward_discount: float,
        target_update_freq: int,
        train_iters: int,
        batch_size: int,
        sampler_cls="uniform",
        sampler_params=None,
        epsilon: float = .0,
        soft_update_coefficient: float = 0.1,
        double: bool = True,
        advantage_type: str = None,
        loss_cls="mse"
    ):
        self.reward_discount = reward_discount
        self.target_update_freq = target_update_freq
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.sampler_cls = get_sampler_cls(sampler_cls)
        self.sampler_params = sampler_params if sampler_params else {}
        self.epsilon = epsilon
        self.soft_update_coefficient = soft_update_coefficient
        self.double = double
        self.advantage_type = advantage_type
        self.loss_func = get_torch_loss_cls(loss_cls)()


class DQN(AbsAlgorithm):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        model (QEstimatorForDiscreteActions): Q-value model.
        config (DQNConfig): Configuration for DQN algorithm.
        experience_memory_size (int): Size of the experience memory. If it is -1, the experience memory is of
            unlimited size.
        experience_memory_overwrite_type (str): A string indicating how experiences in the experience memory are
            to be overwritten after its capacity has been reached. Must be "rolling" or "random".
        empty_experience_memory_after_step (bool): If True, the experience memory will be emptied  after each call
            to ``step``. Defaults to False.
        min_new_experiences_to_trigger_learning (int): Minimum number of new experiences required to trigger learning.
            Defaults to 1.
        min_experiences_to_trigger_learning (int): Minimum number of experiences in the experience memory required for
            training. Defaults to 1.
    """
    def __init__(
        self,
        model: QEstimatorForDiscreteActions,
        config: DQNConfig,
        experience_memory_size: int,
        experience_memory_overwrite_type: str,
        empty_experience_memory_after_step: bool = False,
        min_new_experiences_to_trigger_learning: int = 1,
        min_experiences_to_trigger_learning: int = 1
    ):
        super().__init__(
            model, config, experience_memory_size, experience_memory_overwrite_type,
            empty_experience_memory_after_step,
            min_new_experiences_to_trigger_learning=min_new_experiences_to_trigger_learning,
            min_experiences_to_trigger_learning=min_experiences_to_trigger_learning
        )
        self._sampler = self.config.sampler_cls(self.experience_memory, **self.config.sampler_params)
        self._training_counter = 0
        self._target_model = model.copy() if model.trainable else None
        self._target_model.eval()
        self._num_actions = self.model.num_actions

    def choose_action(self, states) -> Union[int, np.ndarray]:
        with torch.no_grad():
            self.model.eval()
            actions, q_vals = self.model.choose_action(states)

        actions, q_vals = actions.cpu().numpy(), q_vals.cpu().numpy()
        if len(actions) == 1:
            return actions[0] if np.random.random() > self.config.epsilon else np.random.choice(self._num_actions)
        else:
            return np.array([
                action if np.random.random() > self.config.epsilon else np.random.choice(self._num_actions)
                for action in actions
            ])

    def step(self):
        self.model.train()
        for _ in range(self.config.train_iters):
            # sample from the replay memory
            indexes, batch = self._sampler.sample(self.config.batch_size)
            states, next_states = batch["S"], batch["S_"]
            actions = torch.from_numpy(batch["A"])
            rewards = torch.from_numpy(batch["R"])
            q_values = self.model(states, actions)
            # get next Q values
            with torch.no_grad():
                if self.config.double:
                    next_q_values = self._target_model(next_states, self.model.choose_action(next_states)[0])  # (N,)
                else:
                    next_q_values = self._target_model.choose_action(next_states)[1]  # (N,)

            # get TD errors
            target_q_values = (rewards + self.config.gamma * next_q_values).detach()  # (N,)
            loss = self.config.loss_func(q_values, target_q_values)

            # train and update target if necessary 
            self.model.step(loss.mean())
            self._training_counter += 1
            if self._training_counter % self.config.target_update_freq == 0:
                self._target_model.soft_update(self.model, self.config.soft_update_coefficient)

            # update auxillary info for the next round of sampling
            self._sampler.update(indexes, loss.detach().numpy())

    def set_exploration_params(self, epsilon):
        self.config.epsilon = epsilon
