# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple
from typing import Union

import numpy as np
import torch

from maro.rl.model import QNetForDiscreteActionSpace
from maro.rl.utils import get_max, get_sampler_cls, get_td_errors, get_torch_loss_cls, select_by_actions

from .abs_policy import AbsPolicy

DQNExperience = namedtuple("DQNExperience", ["state", "action", "reward", "next_state"])


class DQNConfig:
    """Configuration for the DQN algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        target_update_freq (int): Number of training rounds between target model updates.
        epsilon (float): Exploration rate for epsilon-greedy exploration. Defaults to None.
        soft_update_coefficient (float): Soft update coefficient, e.g.,
            target_model = (soft_update_coefficient) * eval_model + (1-soft_update_coefficient) * target_model.
            Defaults to 1.0.
        double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
        loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class. If it is a string,
            it must be a key in ``TORCH_LOSS``. Defaults to "mse".
    """
    __slots__ = [
        "reward_discount", "target_update_freq", "train_iters", "batch_size", "sampler_cls", "sampler_params",
        "epsilon", "soft_update_coefficient", "double", "loss_func"
    ]

    def __init__(
        self,
        reward_discount: float,
        target_update_freq: int,
        epsilon: float = .0,
        soft_update_coefficient: float = 0.1,
        double: bool = True,
        loss_cls="mse"
    ):
        self.reward_discount = reward_discount
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon
        self.soft_update_coefficient = soft_update_coefficient
        self.double = double
        self.loss_func = get_torch_loss_cls(loss_cls)()


class DQN(AbsPolicy):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        model (QNetForDiscreteActionSpace): Q-value model.
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
    def __init__(self, q_net: QNetForDiscreteActionSpace, config: DQNConfig):
        if not isinstance(q_net, QNetForDiscreteActionSpace):
            raise TypeError("model must be an instance of 'QNetForDiscreteActionSpace'")

        super().__init__(config)
        self.q_net = q_net
        self.target_q_net = model.copy() if model.trainable else None
        self.target_q_net.eval()
        self._training_counter = 0
        self._num_actions = self.q_net.num_actions

    def choose_action(self, states) -> Union[int, np.ndarray]:
        with torch.no_grad():
            self.q_net.eval()
            actions, q_vals = self.q_net.choose_action(states)

        actions, q_vals = actions.cpu().numpy(), q_vals.cpu().numpy()
        if len(actions) == 1:
            return actions[0] if np.random.random() > self.config.epsilon else np.random.choice(self._num_actions)
        else:
            return np.array([
                action if np.random.random() > self.config.epsilon else np.random.choice(self._num_actions)
                for action in actions
            ])

    def update(self, experience_obj: DQNExperience):
        if not isinstance(experience_obj, DQNExperience):
            raise TypeError(f"Expected experience object of type DQNExperience, got {type(experience_obj)}")

        self.q_net.train()       
        # sample from the replay memory
        states, next_states = experience_obj.state, experience_obj.next_state
        actions = torch.from_numpy(experience_obj.action)
        rewards = torch.from_numpy(experience_obj.reward)
        q_values = self.q_net(states, actions)
        # get next Q values
        with torch.no_grad():
            if self.config.double:
                next_q_values = self.target_q_net(next_states, self.q_net.choose_action(next_states)[0])  # (N,)
            else:
                next_q_values = self.target_q_net.choose_action(next_states)[1]  # (N,)

        # get TD errors
        target_q_values = (rewards + self.config.gamma * next_q_values).detach()  # (N,)
        loss = self.config.loss_func(q_values, target_q_values)

        # train and update target if necessary 
        self.q_net.step(loss.mean())
        self._training_counter += 1
        if self._training_counter % self.config.target_update_freq == 0:
            self.target_q_net.soft_update(self.q_net, self.config.soft_update_coefficient)

    def set_exploration_params(self, epsilon):
        self.config.epsilon = epsilon
