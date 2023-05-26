from typing import Any

import torch
from torch.optim import Adam

from maro.rl.model import FullyConnected
from maro.rl_v31.exploration.strategy import GaussianNoise
from maro.rl_v31.model.base import PolicyModel
from maro.rl_v31.model.qnet import QCritic, QNet
from maro.rl_v31.policy import DDPGPolicy
from maro.rl_v31.rl_component_bundle.rl_component_bundle import RLComponentBundle
from maro.rl_v31.training.algorithms.ddpg import DDPGTrainer
from maro.simulator import Env

from tests.rl_v31_gym.gym_wrapper.common import (
    action_limit,
    env,
    env_conf,
    gym_action_dim,
    gym_action_space,
    gym_obs_dim,
    gym_obs_space,
    is_discrete,
    num_agents,
)
from tests.rl_v31_gym.gym_wrapper.env_wrapper import GymEnvWrapper
from tests.rl_v31_gym.gym_wrapper.simulator.business_engine import GymBusinessEngine
from tests.rl_v31_gym.tasks.utils import metrics_agg_func

actor_net_conf = {
    "hidden_dims": [256, 256],
    "activation": torch.nn.ReLU,
    "output_activation": torch.nn.Tanh,
}
critic_net_conf = {
    "hidden_dims": [256, 256],
    "activation": torch.nn.ReLU,
}
actor_learning_rate = 1e-3
critic_learning_rate = 1e-3


class MyPolicyModel(PolicyModel):
    def __init__(self) -> None:
        super().__init__()

        self._net = FullyConnected(
            input_dim=gym_obs_dim,
            output_dim=gym_action_dim,
            hidden_dims=actor_net_conf["hidden_dims"],
            activation=actor_net_conf["activation"],
            output_activation=actor_net_conf["output_activation"],
        )

        self._action_limit = action_limit

    def forward(self, obs: Any) -> torch.Tensor:
        return self._net(obs) * self._action_limit


class MyCriticModel(QNet):
    def __init__(self) -> None:
        super().__init__()

        self.mlp = FullyConnected(
            input_dim=gym_obs_dim + gym_action_dim,
            output_dim=1,
            hidden_dims=critic_net_conf["hidden_dims"],
            activation=critic_net_conf["activation"],
        )

    def q_values(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([obs, act], dim=1).float()).squeeze(-1)


def get_ddpg_policy(name: str) -> DDPGPolicy:
    model = MyPolicyModel()
    optim = Adam(model.parameters(), lr=actor_learning_rate)

    return DDPGPolicy(
        name=name,
        obs_space=gym_obs_space,
        action_space=gym_action_space,
        model=model,
        optim=optim,
        explore_strategy=GaussianNoise(noise_scale=0.1, action_limit=action_limit),
    )


def get_ddpg_critic() -> QCritic:
    model = MyCriticModel()
    optim = Adam(model.parameters(), lr=critic_learning_rate)

    return QCritic(model=model, optim=optim)


assert not is_discrete
agent2policy = {agent: f"ddpg_{agent}.policy" for agent in env.agent_idx_list}
policies = [get_ddpg_policy(f"ddpg_{i}.policy") for i in range(num_agents)]
trainers = [
    DDPGTrainer(
        name=f"ddpg_{i}",
        reward_discount=0.99,
        memory_size=1000000,
        batch_size=100,
        critic_func=get_ddpg_critic,
        num_epochs=50,
        n_start_train=1000,
        soft_update_coef=0.005,
        update_target_every=1,
    )
    for i in range(num_agents)
]


rl_component_bundle = RLComponentBundle(
    env_wrapper_func=lambda: GymEnvWrapper(Env(business_engine_cls=GymBusinessEngine, **env_conf)),
    agent2policy=agent2policy,
    policies=policies,
    trainers=trainers,
    metrics_agg_func=metrics_agg_func,
)
