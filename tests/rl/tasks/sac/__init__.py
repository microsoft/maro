# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from torch.distributions import Normal
from torch.optim import Adam

from maro.rl.model import ContinuousSACNet, QNet
from maro.rl.model.fc_block import FullyConnected
from maro.rl.policy import ContinuousRLPolicy
from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.training.algorithms import SoftActorCriticParams, SoftActorCriticTrainer
from maro.rl.utils import ndarray_to_tensor
from tests.rl.gym_wrapper.common import (
    action_limit,
    action_lower_bound,
    action_upper_bound,
    gym_action_dim,
    gym_action_space,
    gym_state_dim,
    learn_env,
    num_agents,
    test_env,
)
from tests.rl.gym_wrapper.env_sampler import GymEnvSampler

actor_net_conf = {
    "hidden_dims": [256, 256],
    "activation": torch.nn.ReLU,
}
critic_net_conf = {
    "hidden_dims": [256, 256],
    "activation": torch.nn.ReLU,
}
actor_learning_rate = 1e-3
critic_learning_rate = 1e-3

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MyContinuousSACNet(ContinuousSACNet):
    def __init__(self, state_dim: int, action_dim: int, action_limit: float, action_space: spaces.Space) -> None:
        super(MyContinuousSACNet, self).__init__(state_dim=state_dim, action_dim=action_dim)

        self._net = FullyConnected(
            input_dim=state_dim,
            output_dim=actor_net_conf["hidden_dims"][-1],
            hidden_dims=actor_net_conf["hidden_dims"][:-1],
            activation=actor_net_conf["activation"],
            output_activation=actor_net_conf["activation"],
        )
        self._mu = torch.nn.Linear(actor_net_conf["hidden_dims"][-1], action_dim)
        self._log_std = torch.nn.Linear(actor_net_conf["hidden_dims"][-1], action_dim)
        self._action_limit = action_limit
        self._optim = Adam(self.parameters(), lr=actor_learning_rate)

        self._action_space = action_space

    def _get_actions_with_logps_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        net_out = self._net(states.float())
        mu = self._mu(net_out)
        log_std = torch.clamp(self._log_std(net_out), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        pi_action = pi_distribution.rsample() if exploring else mu

        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)

        pi_action = torch.tanh(pi_action) * self._action_limit

        return pi_action, logp_pi

    def _get_random_actions_impl(self, states: torch.Tensor) -> torch.Tensor:
        return torch.stack([ndarray_to_tensor(self._action_space.sample(), device=self._device) for _ in range(states.shape[0])])


class MyQCriticNet(QNet):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(MyQCriticNet, self).__init__(state_dim=state_dim, action_dim=action_dim)
        self._critic = FullyConnected(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=critic_net_conf["hidden_dims"],
            activation=critic_net_conf["activation"],
        )
        self._optim = Adam(self._critic.parameters(), lr=critic_learning_rate)

    def _get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self._critic(torch.cat([states, actions], dim=1).float()).squeeze(-1)


def get_sac_policy(
    name: str,
    action_lower_bound: list,
    action_upper_bound: list,
    gym_state_dim: int,
    gym_action_dim: int,
    action_limit: float,
) -> ContinuousRLPolicy:
    return ContinuousRLPolicy(
        name=name,
        action_range=(action_lower_bound, action_upper_bound),
        policy_net=MyContinuousSACNet(gym_state_dim, gym_action_dim, action_limit, action_space=gym_action_space),
        warmup=10000,
    )


def get_sac_trainer(name: str, state_dim: int, action_dim: int) -> SoftActorCriticTrainer:
    return SoftActorCriticTrainer(
        name=name,
        reward_discount=0.99,
        replay_memory_capacity=1000000,
        batch_size=100,
        params=SoftActorCriticParams(
            get_q_critic_net_func=lambda: MyQCriticNet(state_dim, action_dim),
            update_target_every=1,
            entropy_coef=0.2,
            num_epochs=50,
            n_start_train=1000,
            soft_update_coef=0.005,
        ),
    )


# TODO:
#   1. random seed
#   2. exploration with random sampled action # start_steps=10000,  Number of steps for uniform-random action selection, before running real policy. Helps exploration.
#   3. confirm the effect of (max_ep_len=1000)?

algorithm = "sac"
agent2policy = {agent: f"{algorithm}_{agent}.policy" for agent in learn_env.agent_idx_list}
policies = [
    get_sac_policy(
        f"{algorithm}_{i}.policy",
        action_lower_bound,
        action_upper_bound,
        gym_state_dim,
        gym_action_dim,
        action_limit,
    )
    for i in range(num_agents)
]
trainers = [get_sac_trainer(f"{algorithm}_{i}", gym_state_dim, gym_action_dim) for i in range(num_agents)]

device_mapping = None
if torch.cuda.is_available():
    device_mapping = {f"{algorithm}_{i}.policy": "cuda:0" for i in range(num_agents)}

rl_component_bundle = RLComponentBundle(
    env_sampler=GymEnvSampler(
        learn_env=learn_env,
        test_env=test_env,
        policies=policies,
        agent2policy=agent2policy,
    ),
    agent2policy=agent2policy,
    policies=policies,
    trainers=trainers,
    device_mapping=device_mapping,
)

__all__ = ["rl_component_bundle"]
