# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.algorithm import (
    DDPG, DQN, ActorCritic, ActorCriticConfig, DDPGConfig, DQNConfig, PolicyGradient, PolicyGradientConfig,
    get_rl_policy_cls, get_rl_policy_config_cls, get_rl_policy_model_cls
)
from maro.rl.env_wrapper import AbsEnvWrapper
from maro.rl.experience import AbsExperienceManager, ExperienceSet, Replay, UniformSampler, UseAndDispose
from maro.rl.exploration import (
    AbsExploration, AbsExplorationScheduler, EpsilonGreedyExploration, GaussianNoiseExploration, LinearExplorationScheduler,
    MultiPhaseLinearExplorationScheduler, NoiseExploration, NullExploration, UniformNoiseExploration
)
from maro.rl.model import (
    AbsBlock, AbsCoreModel, FullyConnectedBlock, OptimOption, PolicyNetForDiscreteActionSpace,
    PolicyValueNetForContinuousActionSpace, PolicyValueNetForDiscreteActionSpace, QNetForDiscreteActionSpace
)
from maro.rl.policy import AbsCorePolicy, AbsPolicy, NullPolicy, RLPolicy
from maro.rl.training import (
    Actor, ActorManager, DistributedLearner, EpisodeBasedSchedule, LocalLearner, MultiPolicyUpdateSchedule,
    StepBasedSchedule
)
from maro.rl.utils import (
    get_k_step_returns, get_lambda_returns, get_torch_activation_cls, get_torch_loss_cls, get_torch_lr_scheduler_cls,
    get_torch_optim_cls, get_truncated_cumulative_reward
)

__all__ = [
    "ActorCritic", "ActorCriticConfig", "DDPG", "DDPGConfig", "DQN", "DQNConfig", "PolicyGradient",
    "PolicyGradientConfig", "get_rl_policy_cls", "get_rl_policy_config_cls", "get_rl_policy_model_cls",
    "AbsEnvWrapper",
    "AbsExperienceManager", "ExperienceSet", "Replay", "UniformSampler", "UseAndDispose",
    "AbsExploration", "AbsExplorationScheduler", "EpsilonGreedyExploration", "GaussianNoiseExploration",
    "LinearExplorationScheduler", "MultiPhaseLinearExplorationScheduler", "NoiseExploration", "NullExploration",
    "UniformNoiseExploration",
    "AbsBlock", "AbsCoreModel", "FullyConnectedBlock", "OptimOption", "PolicyNetForDiscreteActionSpace",
    "PolicyValueNetForContinuousActionSpace", "PolicyValueNetForDiscreteActionSpace", "QNetForDiscreteActionSpace",
    "AbsCorePolicy", "AbsPolicy", "NullPolicy", "RLPolicy",
    "Actor", "ActorManager", "DistributedLearner", "EpisodeBasedSchedule", "LocalLearner", "MultiPolicyUpdateSchedule",
    "StepBasedSchedule",
    "get_k_step_returns", "get_lambda_returns", "get_torch_activation_cls", "get_torch_loss_cls",
    "get_torch_lr_scheduler_cls", "get_torch_optim_cls", "get_truncated_cumulative_reward"
]
