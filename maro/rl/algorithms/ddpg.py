# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np
import torch

from maro.rl.experience import ExperienceManager
from maro.rl.model import ContinuousACNet
from maro.rl.policy import AbsCorePolicy
from maro.rl.utils import get_torch_loss_cls


class DDPGConfig:
    """Configuration for the DDPG algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        target_update_freq (int): Number of training rounds between policy target model updates.
        train_epochs (int): Number of training epochs per call to ``update()``. Defaults to 1.
        gradient_iters (int): Number of gradient steps for each mini-batch. Defaults to 1.
        q_value_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for
            the Q-value loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        policy_loss_coefficient (float): The coefficient for policy loss in the total loss function, e.g.,
            loss = q_value_loss + ``policy_loss_coefficient`` * policy_loss. Defaults to 1.0.
        soft_update_coefficient (float): Soft update coefficient, e.g.,
            target_model = (soft_update_coefficient) * eval_model + (1-soft_update_coefficient) * target_model.
            Defaults to 1.0.
    """
    __slots__ = [
        "reward_discount", "target_update_freq", "train_epochs", "gradient_iters", "q_value_loss_func",
        "policy_loss_coefficient", "soft_update_coefficient"
    ]

    def __init__(
        self,
        reward_discount: float,
        target_update_freq: int,
        train_epochs: int = 1,
        gradient_iters: int = 1,
        q_value_loss_cls="mse",
        policy_loss_coefficient: float = 1.0,
        soft_update_coefficient: float = 1.0,
    ):
        self.reward_discount = reward_discount
        self.target_update_freq = target_update_freq
        self.train_epochs = train_epochs
        self.gradient_iters = gradient_iters
        self.q_value_loss_func = get_torch_loss_cls(q_value_loss_cls)()
        self.policy_loss_coefficient = policy_loss_coefficient
        self.soft_update_coefficient = soft_update_coefficient


class DDPG(AbsCorePolicy):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        https://arxiv.org/pdf/1509.02971.pdf
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg

    Args:
        name (str): Policy name.
        ac_net (ContinuousACNet): DDPG policy and q-value models.
        experience_manager (ExperienceManager): An experience manager for storing and retrieving experiences
            for training.
        config (DDPGConfig): Configuration for DDPG algorithm.
        update_trigger (int): Minimum number of new experiences required to trigger an ``update`` call. Defaults to 1.
        warmup (int): Minimum number of experiences in the experience memory required to trigger an ``update`` call.
            Defaults to 1.
    """
    def __init__(
        self,
        name: str,
        ac_net: ContinuousACNet,
        experience_manager: ExperienceManager,
        config: DDPGConfig,
        update_trigger: int = 1,
        warmup: int = 1,
    ):
        if not isinstance(ac_net, ContinuousACNet):
            raise TypeError("model must be an instance of 'ContinuousACNet'")

        super().__init__(name, experience_manager, update_trigger=update_trigger, warmup=warmup)
        self.ac_net = ac_net
        if self.ac_net.trainable:
            self.target_ac_net = ac_net.copy()
            self.target_ac_net.eval()
        else:
            self.target_ac_net = None
        self.config = config
        self.device = self.ac_net.device
        self._train_cnt = 0

    def choose_action(self, states) -> Union[float, np.ndarray]:
        with torch.no_grad():
            actions = self.ac_net.get_action(states).cpu().numpy()

        return actions[0] if len(actions) == 1 else actions

    def update(self):
        self.ac_net.train()
        for _ in range(self.config.train_epochs):
            experience_set = self.experience_manager.get()
            states, next_states = experience_set.states, experience_set.next_states
            actual_actions = torch.from_numpy(experience_set.actions).to(self.device)
            rewards = torch.from_numpy(experience_set.rewards).to(self.device)
            if len(actual_actions.shape) == 1:
                actual_actions = actual_actions.unsqueeze(dim=1)  # (N, 1)

            with torch.no_grad():
                next_q_values = self.target_ac_net.value(next_states)
            target_q_values = (rewards + self.config.reward_discount * next_q_values).detach()  # (N,)

            for _ in range(self.config.gradient_iters):
                q_values = self.ac_net(states, actions=actual_actions).squeeze(dim=1)  # (N,)
                q_value_loss = self.config.q_value_loss_func(q_values, target_q_values)
                policy_loss = -self.ac_net.value(states).mean()
                loss = q_value_loss + self.config.policy_loss_coefficient * policy_loss
                self.ac_net.step(loss)
                self._train_cnt += 1
                if self._train_cnt % self.config.target_update_freq == 0:
                    self.target_ac_net.soft_update(self.ac_net, self.config.soft_update_coefficient)

    def set_state(self, policy_state):
        self.ac_net.load_state_dict(policy_state)
        self.target_ac_net = self.ac_net.copy() if self.ac_net.trainable else None

    def get_state(self):
        return self.ac_net.state_dict()
