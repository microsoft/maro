# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple
from typing import Union

import numpy as np
import torch

from maro.rl.algorithms import AbsAlgorithm
from maro.rl.experience import ExperienceBatch
from maro.rl.exploration import GaussianNoiseExploration
from maro.rl.model import ContinuousACNet
from maro.rl.utils import get_torch_loss_cls

DDPGLossInfo = namedtuple("DDPGLossInfo", ["policy_loss", "q_loss", "total_loss", "grad"])


class DDPG(AbsAlgorithm):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        https://arxiv.org/pdf/1509.02971.pdf
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg

    Args:
        ac_net (ContinuousACNet): DDPG policy and q-value models.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        update_target_every (int): Number of training rounds between policy target model updates.
        q_value_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for
            the Q-value loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        q_value_loss_coeff (float): Coefficient for policy loss in the total loss function, e.g.,
            loss = policy_loss + ``q_value_loss_coeff`` * q_value_loss. Defaults to 1.0.
        soft_update_coeff (float): Soft update coefficient, e.g., target_model = (soft_update_coeff) * eval_model +
            (1-soft_update_coeff) * target_model. Defaults to 1.0.
    """
    def __init__(
        self,
        ac_net: ContinuousACNet,
        reward_discount: float,
        update_target_every: int,
        q_value_loss_cls="mse",
        q_value_loss_coeff: float = 1.0,
        soft_update_coeff: float = 1.0,
        exploration=GaussianNoiseExploration(),
    ):
        if not isinstance(ac_net, ContinuousACNet):
            raise TypeError("model must be an instance of 'ContinuousACNet'")

        super().__init__(exploration=exploration)
        self.ac_net = ac_net
        if self.ac_net.trainable:
            self.target_ac_net = ac_net.copy()
            self.target_ac_net.eval()
        else:
            self.target_ac_net = None
        self.reward_discount = reward_discount
        self.update_target_every = update_target_every
        self.q_value_loss_func = get_torch_loss_cls(q_value_loss_cls)()
        self.q_value_loss_coeff = q_value_loss_coeff
        self.soft_update_coeff = soft_update_coeff
        self.device = self.ac_net.device

        self._ac_net_version = 0
        self._target_ac_net_version = 0

    def choose_action(self, states, explore: bool = False) -> Union[float, np.ndarray]:
        self.ac_net.eval()
        with torch.no_grad():
            actions = self.ac_net.get_action(states).cpu().numpy()

        if explore:
            actions = self.exploration(actions, state=states)
        return actions[0] if len(actions) == 1 else actions

    def apply(self, grad_dict: dict):
        self.ac_net.apply(grad_dict)
        if self._ac_net_version - self._target_ac_net_version == self.update_target_every:
            self._update_target()

    def learn(self, batch: ExperienceBatch, inplace: bool = True):
        assert self.ac_net.trainable, "ac_net needs to have at least one optimizer registered."
        self.ac_net.train()
        states, next_states = batch.data.states, batch.data.next_states
        actual_actions = torch.from_numpy(batch.data.actions).to(self.device)
        rewards = torch.from_numpy(batch.data.rewards).to(self.device)
        if len(actual_actions.shape) == 1:
            actual_actions = actual_actions.unsqueeze(dim=1)  # (N, 1)

        with torch.no_grad():
            next_q_values = self.target_ac_net.value(next_states)
        target_q_values = (rewards + self.reward_discount * next_q_values).detach()  # (N,)

        q_values = self.ac_net(states, actions=actual_actions).squeeze(dim=1)  # (N,)
        q_loss = self.q_value_loss_func(q_values, target_q_values)
        policy_loss = -self.ac_net.value(states).mean()

        total_loss = policy_loss + self.q_value_loss_coeff * q_loss

        if inplace:
            self.ac_net.step(total_loss)
            grad = None
            self._ac_net_version += 1
            if self._ac_net_version - self._target_ac_net_version == self.update_target_every:
                self._update_target()
        else:
            grad = self.ac_net.get_gradients(total_loss)

        return DDPGLossInfo(policy_loss, q_loss, total_loss, grad), batch.indexes

    def _update_target(self):
        # soft-update target network
        self.target_ac_net.soft_update(self.ac_net, self.soft_update_coeff)
        self._target_ac_net_version = self._ac_net_version

    def set_state(self, policy_state):
        self.ac_net.load_state_dict(policy_state)
        self.target_ac_net = self.ac_net.copy() if self.ac_net.trainable else None

    def get_state(self):
        return self.ac_net.state_dict()
