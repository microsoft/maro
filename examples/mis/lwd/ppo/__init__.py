# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.training.algorithms.ppo import PPOParams
from maro.rl.utils.common import get_env

from examples.mis.lwd.ppo.model import GraphBasedPolicyNet, GraphBasedVNet
from examples.mis.lwd.ppo.ppo import GraphBasedPPOPolicy, GraphBasedPPOTrainer


def get_ppo_policy(
    name: str,
    state_dim: int,
    action_num: int,
    hidden_dim: int,
    num_layers: int,
    init_lr: float,
) -> GraphBasedPPOPolicy:
    return GraphBasedPPOPolicy(
        name=name,
        policy_net=GraphBasedPolicyNet(
            state_dim=state_dim,
            action_num=action_num,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            init_lr=init_lr,
        ),
    )


def get_ppo_trainer(
    name: str,
    state_dim: int,
    hidden_dim: int,
    num_layers: int,
    init_lr: float,
    clip_ratio: float,
    max_tick: int,
    batch_size: int,
    reward_discount: float,
    graph_batch_size: int,
    graph_num_samples: int,
    num_train_epochs: int,
    norm_base: float,
) -> GraphBasedPPOTrainer:
    return GraphBasedPPOTrainer(
        name=name,
        params=PPOParams(
            get_v_critic_net_func=lambda: GraphBasedVNet(state_dim, hidden_dim, num_layers, init_lr, norm_base),
            grad_iters=1,
            lam=None,  # GAE not used here.
            clip_ratio=clip_ratio,
        ),
        replay_memory_capacity=max_tick,
        batch_size=batch_size,
        reward_discount=reward_discount,
        graph_batch_size=graph_batch_size,
        graph_num_samples=graph_num_samples,
        input_feature_size=state_dim,
        num_train_epochs=num_train_epochs,
        log_dir=get_env("LOG_PATH", required=False, default=None),
    )


__all__ = ["get_ppo_policy", "get_ppo_trainer"]
