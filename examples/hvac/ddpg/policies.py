# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from itertools import chain
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from maro.rl.exploration import MultiLinearExplorationScheduler
from maro.rl.modeling import ContinuousACNet, ContinuousSACNet, FullyConnected
from maro.rl.policy import DDPG, SoftActorCritic

from .config import (
    ac_net_config, action_dim, action_lower_bound, action_upper_bound, algorithm, ddpg_config, sac_policy_net_config,
    sac_policy_net_optim_config, sac_q_net_config, sac_q_net_optim_config, sac_config, state_dim
)


class AhuACNet(ContinuousACNet):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_lower_bound: List[float],
        output_upper_bound: List[float],
        actor_hidden_dims: List[int],
        critic_hidden_dims: List[int],
        actor_activation,
        critic_activation,
        actor_optimizer,
        critic_optimizer,
        actor_lr: float,
        critic_lr: float
    ):
        super().__init__(
            out_min=output_lower_bound,
            out_max=output_upper_bound
        )

        self._input_dim = input_dim
        self._output_dim = output_dim

        self._base_value = torch.tensor(output_lower_bound)
        self._gap_size = torch.tensor(output_upper_bound) - torch.tensor(output_lower_bound)

        self.actor = nn.Sequential(
            FullyConnected(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=actor_hidden_dims,
                activation=actor_activation,
                head=True
            ),
            nn.Tanh()
        )
        self.actor_optimizer = actor_optimizer(
            params=self.actor.parameters(),
            lr=actor_lr
        )

        self.critic = FullyConnected(
            input_dim=input_dim + output_dim,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=critic_activation,
            head=True
        )
        self.critic_optimizer = critic_optimizer(
            params=self.critic.parameters(),
            lr=critic_lr
        )

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def action_dim(self):
        return self._output_dim

    def forward(self, states: torch.tensor, actions: torch.tensor=None) -> torch.tensor:
        if actions is None:
            return (self.actor(states) + 1) / 2 * self._gap_size + self._base_value
        else:
            return self.critic(torch.cat([states, actions], dim=1))

    def step(self, loss: torch.tensor):
        # TODO: to be confirmed
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def set_state(self, state):
        self.load_state_dict(state["network"]),
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])

    def get_state(self):
        return {
            "network": self.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict()
        }


def relative_gaussian_noise(
    state: np.ndarray,
    action: np.ndarray,
    min_action: Union[float, list, np.ndarray],
    max_action: Union[float, list, np.ndarray],
    mean: Union[float, list, np.ndarray] = .0,
    stddev: Union[float, list, np.ndarray] = 1.0,
) -> Union[float, np.ndarray]:
    noise = np.random.normal(loc=mean, scale=stddev, size=action.shape)
    action = action + noise * (np.array(max_action) - np.array(min_action))
    return np.clip(action, min_action, max_action)


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class AhuSACNet(ContinuousSACNet):
    def __init__(self):
        super().__init__()
        # policy net
        self.policy_base = FullyConnected(**sac_policy_net_config)
        self.mu_layer = nn.Linear(self.policy_base.output_dim, action_dim)
        self.log_std_layer = nn.Linear(self.policy_base.output_dim, action_dim)
        self.policy_optim = sac_policy_net_optim_config[0](self.policy_params, **sac_policy_net_optim_config[1])

        # Q nets
        self.q1 = FullyConnected(**sac_q_net_config)
        self.q2 = FullyConnected(**sac_q_net_config)
        self.logp_softplus = nn.Softplus()
        self.q_optim = sac_q_net_optim_config[0](self.q_params, **sac_q_net_optim_config[1])

        self._min_action = torch.from_numpy(action_lower_bound)
        self._max_action = torch.from_numpy(action_upper_bound)

    @property
    def input_dim(self):
        return state_dim

    @property
    def action_dim(self):
        return action_dim

    @property
    def action_min(self):
        return action_lower_bound

    @property
    def action_max(self):
        return action_upper_bound

    @property
    def policy_params(self):
        return chain(self.policy_base.parameters(), self.mu_layer.parameters(), self.log_std_layer.parameters())

    @property
    def q_params(self):
        return chain(self.q1.parameters(), self.q2.parameters())

    def forward(self, states, deterministic=False):
        base_out = self.policy_base(states)
        mu = self.mu_layer(base_out)
        log_std = torch.clamp(self.log_std_layer(base_out), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        distribution = Normal(mu, std)
        action = mu if deterministic else distribution.rsample()

        logp_pi = (
            distribution.log_prob(action).sum(axis=-1) -
            (2 * (np.log(2) - action - self.logp_softplus(-2 * action))).sum(axis=1)
        )

        action = self._min_action + (self._max_action - self._min_action) * torch.tanh(action)
        return action.float(), logp_pi.float()

    def get_q1_values(self, states: torch.tensor, actions: torch.tensor):
        inputs = torch.cat([states, actions], dim=-1)
        #print(inputs)
        return self.q1(inputs).squeeze(dim=-1)

    def get_q2_values(self, states: torch.tensor, actions: torch.tensor):
        inputs = torch.cat([states, actions], dim=-1)
        #print(inputs)
        return self.q2(inputs).squeeze(dim=-1)

    def step(self, loss):
        self.policy_optim.zero_grad()
        self.q_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()
        self.q_optim.step()

    def get_state(self):
        return {
            "network": self.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
            "q_optim": self.q_optim.state_dict()
        }

    def set_state(self, state):
        self.load_state_dict(state["network"])
        self.policy_optim.load_state_dict(state["policy_optim"])
        self.q_optim.load_state_dict(state["q_optim"])


if algorithm == "ddpg":
    policy_func_dict = {
        "ddpg": lambda name: DDPG(
            name=name,
            ac_net=AhuACNet(**ac_net_config),
            reward_discount=0.99,
            warmup=5000,
            exploration_strategy=(relative_gaussian_noise, ddpg_config["exploration_strategy"]),
            exploration_scheduling_options=[
                ("mean", MultiLinearExplorationScheduler, ddpg_config["exploration_mean_scheduler_options"]),
            ],
            train_batch_size=256,
        )
    }
elif algorithm == "sac":
    policy_func_dict = {
        "sac": lambda name: SoftActorCritic(
            name=name,
            sac_net=AhuSACNet(),
            **sac_config
        )
    }
else:
    raise ValueError(f"Unsupported algorithm: {algorithm}. Supported algorithms are 'ddpg' and 'sac'.")
