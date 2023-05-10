# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
import torch
from torch.optim import Adam

from maro.rl.model import DiscreteQNet, FullyConnected
from maro.rl.policy import ValueBasedPolicy
from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.training.algorithms import DQNParams, DQNTrainer

from tests.rl.gym_wrapper.common import gym_action_num, gym_state_dim, is_discrete, learn_env, num_agents, test_env
from tests.rl.gym_wrapper.env_sampler import GymEnvSampler

net_conf = {
    "hidden_dims": [256],
    "activation": torch.nn.ReLU,
    "output_activation": None,
}
lr = 1e-3


class LinearExplore:
    def __init__(self) -> None:
        self._call_count = 0

    def explore_func(
        self,
        state: np.ndarray,
        action: np.ndarray,
        num_actions: int,
        *,
        explore_steps: int,
        start_explore_prob: float,
        end_explore_prob: float,
    ) -> np.ndarray:
        ratio = min(self._call_count / explore_steps, 1.0)
        epsilon = start_explore_prob + (end_explore_prob - start_explore_prob) * ratio
        explore_flag = np.random.random() < epsilon
        action = np.array([np.random.randint(num_actions) if explore_flag else act for act in action])

        self._call_count += 1
        return action


linear_explore = LinearExplore()


class MyQNet(DiscreteQNet):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(MyQNet, self).__init__(state_dim=state_dim, action_num=action_num)

        self._mlp = FullyConnected(
            input_dim=state_dim,
            output_dim=action_num,
            **net_conf,
        )
        self._optim = Adam(self._mlp.parameters(), lr=lr)

    def _get_q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        return self._mlp(states)


def get_dqn_policy(
    name: str,
    state_dim: int,
    action_num: int,
) -> ValueBasedPolicy:
    return ValueBasedPolicy(
        name=name,
        q_net=MyQNet(state_dim=state_dim, action_num=action_num),
        exploration_strategy=(
            linear_explore.explore_func,
            {
                "explore_steps": 10000,
                "start_explore_prob": 1.0,
                "end_explore_prob": 0.02,
            },
        ),
        warmup=0,  # TODO: check this
    )


def get_dqn_trainer(
    name: str,
) -> DQNTrainer:
    return DQNTrainer(
        name=name,
        params=DQNParams(
            use_prioritized_replay=False,  #
            # alpha=0.4,
            # beta=0.6,
            num_epochs=50,
            update_target_every=10,
            soft_update_coef=1.0,
        ),
        replay_memory_capacity=50000,
        batch_size=64,
        reward_discount=1.0,
    )


assert is_discrete

algorithm = "dqn"
agent2policy = {agent: f"{algorithm}_{agent}.policy" for agent in learn_env.agent_idx_list}
policies = [
    get_dqn_policy(
        f"{algorithm}_{i}.policy",
        state_dim=gym_state_dim,
        action_num=gym_action_num,
    )
    for i in range(num_agents)
]
trainers = [get_dqn_trainer(f"{algorithm}_{i}") for i in range(num_agents)]

device_mapping = {f"{algorithm}_{i}.policy": "cuda:0" for i in range(num_agents)} if torch.cuda.is_available() else None

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
