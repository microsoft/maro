import torch
from torch.optim import Adam, RMSprop

from examples.cim.rl.config import action_num, env_conf, num_agents, reward_shaping_conf, state_dim
from examples.cim.rl.env_sampler import CIMEnvSampler
from maro.easyrl.policy import PPOPolicy
from maro.easyrl.rollout import SimpleEasyRLJob
from maro.rl.model import DiscreteACBasedNet, FullyConnected, VNet
from maro.rl.policy import DiscretePolicyGradient
from maro.simulator import Env


class MyActorNet(DiscreteACBasedNet):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(MyActorNet, self).__init__(state_dim=state_dim, action_num=action_num)
        self._actor = FullyConnected(
            input_dim=state_dim,
            output_dim=action_num,
            hidden_dims=[256, 128, 64],
            activation=torch.nn.Tanh,
            softmax=True,
            batch_norm=False,
            head=True,
        )
        self._optim = Adam(self._actor.parameters(), lr=0.001)

    def _get_action_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        return self._actor(states)


class MyCriticNet(VNet):
    def __init__(self, state_dim: int) -> None:
        super(MyCriticNet, self).__init__(state_dim=state_dim)
        self._critic = FullyConnected(
            input_dim=state_dim,
            hidden_dims=[256, 128, 64],
            output_dim=1,
            activation=torch.nn.LeakyReLU,
            softmax=False,
            batch_norm=True,
            head=True,
        )
        self._optim = RMSprop(self._critic.parameters(), lr=0.001)

    def _get_v_values(self, states: torch.Tensor) -> torch.Tensor:
        return self._critic(states).squeeze(-1)


if __name__ == "__main__":
    env = Env(**env_conf)
    env_sampler = CIMEnvSampler(learn_env=env, test_env=env, reward_eval_delay=reward_shaping_conf["time_window"])

    agent_policy_dict = {}
    for i in range(num_agents):
        agent_policy_dict[i] = PPOPolicy(
            actor=DiscretePolicyGradient(name=f"ppo_{i}", policy_net=MyActorNet(state_dim, action_num)),
            critic=MyCriticNet(state_dim),
            clip_ratio=0.1,
            reward_discount=.0,
            grad_iters=10,
            critic_loss_cls=torch.nn.SmoothL1Loss,
            min_logp=None,
            lam=.0,
        )

    rl_job = SimpleEasyRLJob(env_sampler=env_sampler, agent_policy_dict=agent_policy_dict)
    rl_job.train(30)
