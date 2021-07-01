# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.algorithms import (
    DDPG, DQN, ActorCritic, ActorCriticConfig, DDPGConfig, DQNConfig, PolicyGradient, PolicyGradientConfig,
    get_rl_policy_cls, get_rl_policy_config_cls, get_rl_policy_model_cls
)
from maro.rl.asynchronous import actor, policy_server
from maro.rl.early_stopping import AbsEarlyStopper
from maro.rl.experience import AbsSampler, ExperienceManager, ExperienceSet, PrioritizedSampler
from maro.rl.exploration import (
    AbsExploration, AbsExplorationScheduler, EpsilonGreedyExploration, GaussianNoiseExploration,
    LinearExplorationScheduler, MultiPhaseLinearExplorationScheduler, NoiseExploration, NullExploration,
    UniformNoiseExploration
)
from maro.rl.local import LocalLearner
from maro.rl.model import (
    AbsBlock, AbsCoreModel, ContinuousACNet, DiscreteACNet, DiscretePolicyNet, DiscreteQNet, FullyConnectedBlock,
    OptimOption
)
from maro.rl.policy import (
    AbsCorePolicy, AbsPolicy, AbsPolicyManager, LocalPolicyManager, MultiNodePolicyManager, MultiProcessPolicyManager,
    NullPolicy, trainer_node, trainer_process
)
from maro.rl.synchronous import (
    AbsRolloutManager, Learner, LocalRolloutManager, MultiNodeRolloutManager, MultiProcessRolloutManager,
    rollout_worker_node, rollout_worker_process
)
from maro.rl.utils import (
    MsgKey, MsgTag, get_k_step_returns, get_lambda_returns, get_torch_activation_cls, get_torch_loss_cls,
    get_torch_lr_scheduler_cls, get_torch_optim_cls, get_truncated_cumulative_reward
)
from maro.rl.wrappers import AbsEnvWrapper, AgentWrapper

__all__ = [
    "ActorCritic", "ActorCriticConfig", "DDPG", "DDPGConfig", "DQN", "DQNConfig", "PolicyGradient",
    "PolicyGradientConfig", "get_rl_policy_cls", "get_rl_policy_config_cls", "get_rl_policy_model_cls",
    "actor", "policy_server",
    "AbsEarlyStopper",
    "AbsSampler", "ExperienceManager", "ExperienceSet", "PrioritizedSampler",
    "AbsExploration", "AbsExplorationScheduler", "EpsilonGreedyExploration", "GaussianNoiseExploration",
    "LinearExplorationScheduler", "MultiPhaseLinearExplorationScheduler", "NoiseExploration", "NullExploration",
    "UniformNoiseExploration",
    "LocalLearner",
    "AbsBlock", "AbsCoreModel", "ContinuousACNet", "DiscreteACNet", "DiscretePolicyNet", "DiscreteQNet",
    "FullyConnectedBlock", "OptimOption",
    "AbsCorePolicy", "AbsPolicy", "AbsPolicyManager", "LocalPolicyManager", "MultiNodePolicyManager",
    "MultiProcessPolicyManager", "NullPolicy", "trainer_node", "trainer_process",
    "AbsRolloutManager", "Learner", "LocalRolloutManager", "MultiNodeRolloutManager", "MultiProcessRolloutManager",
    "rollout_worker_node", "rollout_worker_process",
    "MsgKey", "MsgTag", "get_k_step_returns", "get_lambda_returns", "get_torch_activation_cls", "get_torch_loss_cls",
    "get_torch_lr_scheduler_cls", "get_torch_optim_cls", "get_truncated_cumulative_reward",
    "AbsEnvWrapper", "AgentWrapper"
]
