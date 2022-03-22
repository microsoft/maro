import torch

from maro.rl.policy import DiscretePolicyGradient
from maro.rl.training.algorithms import DiscretePPOParams, DiscretePPOTrainer

from .ac import MyActorNet, MyCriticNet


def get_policy(state_dim: int, action_num: int, name: str) -> DiscretePolicyGradient:
    return DiscretePolicyGradient(name=name, policy_net=MyActorNet(state_dim, action_num))


def get_ppo(state_dim: int, name: str) -> DiscretePPOTrainer:
    return DiscretePPOTrainer(
        name=name,
        params=DiscretePPOParams(
            device="cpu",
            get_v_critic_net_func=lambda: MyCriticNet(state_dim),
            reward_discount=.0,
            grad_iters=10,
            critic_loss_cls=torch.nn.SmoothL1Loss,
            min_logp=None,
            lam=.0,
            clip_ratio=0.1,
        ),
    )
