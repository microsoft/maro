# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.optim import Adam

from maro.rl.model import QNet
from maro.rl.model.algorithm_nets.ddpg import ContinuousDDPGNet
from maro.rl.model.fc_block import FullyConnected
from maro.rl.policy import ContinuousRLPolicy
from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.training.algorithms import DDPGParams, DDPGTrainer

from tests.rl.gym_wrapper.common import (
    action_limit,
    action_lower_bound,
    action_upper_bound,
    gym_action_dim,
    gym_state_dim,
    learn_env,
    num_agents,
    test_env,
)
from tests.rl.gym_wrapper.env_sampler import GymEnvSampler

actor_net_conf = {
    "hidden_dims": [256, 256],
    "activation": torch.nn.ReLU,
    "output_activation": torch.nn.Tanh,
}
critic_net_conf = {
    "hidden_dims": [256, 256],
    "activation": torch.nn.Tanh,
}
actor_learning_rate = 1e-3
critic_learning_rate = 1e-3


class MyContinuousDDPGNet(ContinuousDDPGNet):
    def __init__(self, state_dim: int, action_dim: int, action_limit: float) -> None:
        super(MyContinuousDDPGNet, self).__init__(state_dim=state_dim, action_dim=action_dim)

        self._net = FullyConnected(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=actor_net_conf["hidden_dims"],
            activation=actor_net_conf["activation"],
            output_activation=actor_net_conf["output_activation"],
        )
        self._optim = Adam(self._net.parameters(), lr=critic_learning_rate)
        self._action_limit = action_limit
        self._noise_scale = 0.1  # TODO

    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        action = self._net(states) * self._action_limit
        if exploring:
            action += torch.randn(self.action_dim) * self._noise_scale
            action = torch.clamp(action, -self._action_limit, self._action_limit)
        return action


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


def get_ddpg_policy(
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
        policy_net=MyContinuousDDPGNet(gym_state_dim, gym_action_dim, action_limit),
    )


def get_ddpg_trainer(name: str, state_dim: int, action_dim: int) -> DDPGTrainer:
    return DDPGTrainer(
        name=name,
        reward_discount=0.99,
        replay_memory_capacity=1000000,
        batch_size=100,
        params=DDPGParams(
            get_q_critic_net_func=lambda: MyQCriticNet(state_dim, action_dim),
            num_epochs=20,
            n_start_train=1000,
            soft_update_coef=0.005,
        ),
    )


algorithm = "ddpg"
agent2policy = {agent: f"{algorithm}_{agent}.policy" for agent in learn_env.agent_idx_list}
policies = [
    get_ddpg_policy(
        f"{algorithm}_{i}.policy",
        action_lower_bound,
        action_upper_bound,
        gym_state_dim,
        gym_action_dim,
        action_limit,
    )
    for i in range(num_agents)
]
trainers = [get_ddpg_trainer(f"{algorithm}_{i}", gym_state_dim, gym_action_dim) for i in range(num_agents)]

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
)

__all__ = ["rl_component_bundle"]
