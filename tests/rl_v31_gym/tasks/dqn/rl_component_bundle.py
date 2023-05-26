import torch
from torch.optim import Adam

from maro.rl.model import FullyConnected
from maro.rl_v31.exploration.strategy import LinearExploration
from maro.rl_v31.model.qnet import DiscreteQNet
from maro.rl_v31.policy.dqn import DQNPolicy
from maro.rl_v31.rl_component_bundle.rl_component_bundle import RLComponentBundle
from maro.rl_v31.training.algorithms.dqn import DQNTrainer
from maro.simulator import Env

from tests.rl_v31_gym.gym_wrapper.common import (
    env,
    env_conf,
    gym_action_num,
    gym_action_space,
    gym_obs_dim,
    gym_obs_space,
    is_discrete,
    num_agents,
)
from tests.rl_v31_gym.gym_wrapper.env_wrapper import GymEnvWrapper
from tests.rl_v31_gym.gym_wrapper.simulator.business_engine import GymBusinessEngine
from tests.rl_v31_gym.tasks.utils import metrics_agg_func

net_conf = {
    "hidden_dims": [256],
    "activation": torch.nn.ReLU,
    "output_activation": None,
}
lr = 1e-3


class MyQNet(DiscreteQNet):
    def __init__(self) -> None:
        super().__init__()

        self._mlp = FullyConnected(
            input_dim=gym_obs_dim,
            output_dim=gym_action_num,
            **net_conf,
        )

    def q_values_for_all(self, obs: torch.Tensor) -> torch.Tensor:
        return self._mlp(obs)


def get_dqn_policy(name: str) -> DQNPolicy:
    qnet = MyQNet()
    optim = Adam(qnet.parameters(), lr=lr)

    return DQNPolicy(
        name=name,
        obs_space=gym_obs_space,
        action_space=gym_action_space,
        optim=optim,
        qnet=qnet,
        explore_strategy=LinearExploration(
            num_actions=gym_action_num,
            explore_steps=10000,
            start_explore_prob=1.0,
            end_explore_prob=0.02,
        ),
    )


assert is_discrete
agent2policy = {agent: f"dqn_{agent}.policy" for agent in env.agent_idx_list}
policies = [get_dqn_policy(f"dqn_{i}.policy") for i in range(num_agents)]
trainers = [
    DQNTrainer(
        name=f"dqn_{i}",
        memory_size=50000,
        batch_size=64,
        reward_discount=1.0,
        prioritized_params=(0.4, 0.6),
        num_epochs=50,
        update_target_every=10,
        soft_update_coef=1.0,
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
