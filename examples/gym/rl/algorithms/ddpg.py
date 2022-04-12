from typing import Dict, Tuple

import torch
from torch.optim import Adam

from examples.gym.rl.algorithms.utils import mlp
from examples.gym.rl.env_helper import action_lower_bound, action_upper_bound, gym_action_dim, gym_env, gym_state_dim
from maro.rl.model import ContinuousDDPGNet, ContinuousQNet
from maro.rl.policy import ContinuousRLPolicy
from maro.rl.training.algorithms import DDPGParams, DDPGTrainer

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


class MyActorNet(ContinuousDDPGNet):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(MyActorNet, self).__init__(state_dim=state_dim, action_dim=action_dim)

        self._fc = mlp(
            [state_dim] + actor_net_conf["hidden_dims"] + [action_dim],
            activation=actor_net_conf["activation"],
            output_activation=torch.nn.Tanh,
        )
        self._limit = gym_env.action_space.high[0]
        self._optim = Adam(self._fc.parameters(), lr=actor_learning_rate)

    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        actions = self._limit * self._fc(states.float())
        actions += 0.1 * torch.rand(actions.shape)
        return torch.clamp(actions, -self._limit, self._limit)


class MyCriticNet(ContinuousQNet):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(MyCriticNet, self).__init__(state_dim=state_dim, action_dim=action_dim)
        self._critic = mlp(
            [state_dim + action_dim] + critic_net_conf["hidden_dims"] + [1],
            activation=critic_net_conf["activation"],
        )
        self._optim = Adam(self._critic.parameters(), lr=critic_learning_rate)

    def _get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self._critic(torch.cat([states.float(), actions], dim=1)).reshape(-1)


def get_policy(name: str) -> ContinuousRLPolicy:
    return ContinuousRLPolicy(
        name=name,
        action_range=(action_lower_bound, action_upper_bound),
        policy_net=MyActorNet(gym_state_dim, gym_action_dim),
    )


def get_ddpg(name: str) -> DDPGTrainer:
    return DDPGTrainer(
        name=name,
        params=DDPGParams(
            replay_memory_capacity=int(1e6),
            batch_size=128,
            get_q_critic_net_func=lambda: MyCriticNet(gym_state_dim, gym_action_dim),
            reward_discount=0.99,
            soft_update_coef=0.01,
            update_target_every=1,
            num_epochs=100,
            n_start_train=10000,
        ),
    )
