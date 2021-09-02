# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import numpy as np
import torch

from maro.rl.exploration import GaussianNoiseExploration, NoiseExploration
from maro.rl.modeling import ContinuousACNet
from maro.rl.utils import get_torch_loss_cls

from .policy import RLPolicy
from .replay import ReplayMemory


class DDPG(RLPolicy):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        https://arxiv.org/pdf/1509.02971.pdf
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg

    Args:
        name (str): Unique identifier for the policy.
        ac_net (ContinuousACNet): DDPG policy and q-value models.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        update_target_every (int): Number of training rounds between policy target model updates.
        q_value_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for
            the Q-value loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        q_value_loss_coeff (float): Coefficient for policy loss in the total loss function, e.g.,
            loss = policy_loss + ``q_value_loss_coeff`` * q_value_loss. Defaults to 1.0.
        soft_update_coeff (float): Soft update coefficient, e.g., target_model = (soft_update_coeff) * eval_model +
            (1-soft_update_coeff) * target_model. Defaults to 1.0.
        exploration: Exploration strategy for generating exploratory actions. Defaults to ``GaussianNoiseExploration``.
        replay_memory_capacity (int): Capacity of the replay memory. Defaults to 10000.
        random_overwrite (bool): This specifies overwrite behavior when the replay memory capacity is reached. If True,
            overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
            wrap-around. Defaults to False.
        batch_size (int): Training sample. Defaults to 32.
    """
    def __init__(
        self,
        name: str,
        ac_net: ContinuousACNet,
        reward_discount: float,
        num_epochs: int = 1,
        update_target_every: int = 5,
        q_value_loss_cls="mse",
        q_value_loss_coeff: float = 1.0,
        soft_update_coeff: float = 1.0,
        exploration: NoiseExploration = GaussianNoiseExploration(),
        replay_memory_capacity: int = 10000,
        random_overwrite: bool = False,
        batch_size: int = 32
    ):
        if not isinstance(ac_net, ContinuousACNet):
            raise TypeError("model must be an instance of 'ContinuousACNet'")

        super().__init__(name)
        self.ac_net = ac_net
        self.device = self.ac_net.device
        if self.ac_net.trainable:
            self.target_ac_net = ac_net.copy()
            self.target_ac_net.eval()
        else:
            self.target_ac_net = None
        self.reward_discount = reward_discount
        self.num_epochs = num_epochs
        self.update_target_every = update_target_every
        self.q_value_loss_func = get_torch_loss_cls(q_value_loss_cls)()
        self.q_value_loss_coeff = q_value_loss_coeff
        self.soft_update_coeff = soft_update_coeff

        self._ac_net_version = 0
        self._target_ac_net_version = 0

        self._replay_memory = ReplayMemory(
            replay_memory_capacity, self.ac_net.input_dim, action_dim=1, random_overwrite=random_overwrite
        )
        self.batch_size = batch_size

        self.exploration = exploration
        self.greedy = True

    def choose_action(self, states) -> Union[float, np.ndarray]:
        self.ac_net.eval()
        with torch.no_grad():
            actions = self.ac_net.get_action(states).cpu().numpy()

        if not self.greedy:
            actions = self.exploration(actions, state=states)
        return actions[0] if len(actions) == 1 else actions

    def record(
        self,
        key: str,
        state: np.ndarray,
        action: Union[int, float, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        terminal: bool
    ):
        if next_state is None:
            next_state = np.zeros(state.shape, dtype=np.float32)

        self._replay_memory.put(
            np.expand_dims(state, axis=0),
            np.expand_dims(action, axis=0),
            np.expand_dims(reward, axis=0),
            np.expand_dims(next_state, axis=0),
            np.expand_dims(terminal, axis=0)
        )

    def get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> dict:
        assert self.ac_net.trainable, "ac_net needs to have at least one optimizer registered."
        self.ac_net.train()
        states = torch.from_numpy(batch["states"]).to(self.device)
        next_states = torch.from_numpy(["next_states"]).to(self.device)
        actual_actions = torch.from_numpy(batch["actions"]).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        terminals = torch.from_numpy(batch["terminals"]).float().to(self.device)
        if len(actual_actions.shape) == 1:
            actual_actions = actual_actions.unsqueeze(dim=1)  # (N, 1)

        with torch.no_grad():
            next_q_values = self.target_ac_net.value(next_states)
        target_q_values = (rewards + self.reward_discount * (1 - terminals) * next_q_values).detach()  # (N,)

        # loss info
        loss_info = {}
        q_values = self.ac_net(states, actions=actual_actions).squeeze(dim=1)  # (N,)
        q_loss = self.q_value_loss_func(q_values, target_q_values)
        policy_loss = -self.ac_net.value(states).mean()
        loss = policy_loss + self.q_value_loss_coeff * q_loss
        loss_info = {
            "policy_loss": policy_loss.detach().cpu().numpy(),
            "q_loss": q_loss.detach().cpu().numpy(),
            "loss": loss.detach().cpu().numpy()
        }
        if explicit_grad:
            loss_info["grad"] = self.ac_net.get_gradients(loss)

        return loss_info

    def update(self, loss_info_list: List[dict]):
        self.ac_net.apply_gradients([loss_info["grad"] for loss_info in loss_info_list])
        if self._ac_net_version - self._target_ac_net_version == self.update_target_every:
            self._update_target()

    def learn(self, batch: dict):
        self._replay_memory.put(
            batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["terminals"]
        )

        for _ in range(self.num_epochs):
            train_batch = self._replay_memory.sample(self.batch_size)
            self.ac_net.step(self.get_batch_loss(train_batch)["loss"])
            self._ac_net_version += 1
            if self._ac_net_version - self._target_ac_net_version == self.update_target_every:
                self._update_target()

    def _update_target(self):
        # soft-update target network
        self.target_ac_net.soft_update(self.ac_net, self.soft_update_coeff)
        self._target_ac_net_version = self._ac_net_version

    @property
    def exploration_params(self):
        return self.exploration.parameters

    def exploit(self):
        self.greedy = True

    def explore(self):
        self.greedy = False

    def exploration_step(self):
        self.exploration.step()

    def set_state(self, policy_state):
        self.ac_net.load_state_dict(policy_state)
        self.target_ac_net = self.ac_net.copy() if self.ac_net.trainable else None

    def get_state(self):
        return self.ac_net.state_dict()
