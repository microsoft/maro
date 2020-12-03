# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.distributed.learner_actor.abs_actor import AbsActor
from maro.rl.agent.abs_agent import AbsAgent
from maro.rl.agent.abs_agent_manager import AbsAgentManager, AgentManagerMode
from maro.rl.agent.simple_agent_manager import SimpleAgentManager
from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.algorithms.dqn import DQN, DQNConfig, DuelingDQNTask
from maro.rl.algorithms.utils import preprocess, to_device, validate_task_names
from maro.rl.distributed.actor_trainer.abs_autonomous_actor import AbsAutoActor
from maro.rl.distributed.actor_trainer.autonomous_actor import SimpleAutoActor, SEEDAutoActor
from maro.rl.distributed.actor_trainer.trainer import SEEDTrainer, Trainer
from maro.rl.distributed.learner_actor.abs_actor import AbsActor
from maro.rl.distributed.learner_actor.abs_dist_learner import AbsDistLearner
from maro.rl.distributed.learner_actor.actor import SEEDActor, SimpleActor
from maro.rl.distributed.common import ActorTrainerComponent, LearnerActorComponent
from maro.rl.distributed.learner_actor.dist_learner import SEEDLearner, SimpleDistLearner
from maro.rl.distributed.experience_collection import (
    concat_experiences_by_agent, merge_experiences_with_trajectory_boundaries
)
from maro.rl.exploration.abs_explorer import AbsExplorer
from maro.rl.exploration.epsilon_greedy_explorer import EpsilonGreedyExplorer
from maro.rl.learner.abs_learner import AbsLearner
from maro.rl.learner.simple_learner import SimpleLearner
from maro.rl.models.abs_block import AbsBlock
from maro.rl.models.fc_block import FullyConnectedBlock
from maro.rl.models.learning_model import LearningModule, LearningModuleManager, OptimizerOptions
from maro.rl.scheduling.exploration_parameter_generator import (
    DynamicExplorationParameterGenerator, LinearExplorationParameterGenerator, StaticExplorationParameterGenerator,
    TwoPhaseLinearExplorationParameterGenerator
)
from maro.rl.scheduling.scheduler import Scheduler
from maro.rl.shaping.abs_shaper import AbsShaper
from maro.rl.shaping.action_shaper import ActionShaper
from maro.rl.shaping.experience_shaper import ExperienceShaper
from maro.rl.shaping.k_step_experience_shaper import KStepExperienceShaper
from maro.rl.shaping.state_shaper import StateShaper
from maro.rl.storage.abs_store import AbsStore
from maro.rl.storage.column_based_store import ColumnBasedStore
from maro.rl.storage.utils import OverwriteType

__all__ = [
    'AbsActor',
    'AbsAgent',
    'AbsAgentManager',
    'AbsAlgorithm',
    'AbsAutoActor',
    'AbsDistLearner',
    'AbsExplorer',
    'AbsLearner',
    'AbsShaper',
    'AbsStore',
    'ActionShaper',
    'ActorTrainerComponent',
    'AgentManagerMode',
    'ColumnBasedStore',
    'DQN',
    'DQNConfig',
    'DuelingDQNTask',
    'DynamicExplorationParameterGenerator',
    'EpsilonGreedyExplorer',
    'ExperienceShaper',
    'FullyConnectedBlock',
    'KStepExperienceShaper',
    'LearnerActorComponent',
    'LearningModuleManager',
    'LearningModule',
    'LinearExplorationParameterGenerator',
    'OptimizerOptions',
    'OverwriteType',
    'SEEDActor',
    'SEEDAutoActor',
    'SEEDLearner',
    'SEEDTrainer',
    'Scheduler',
    'SimpleActor',
    'SimpleAgentManager',
    'SimpleAutoActor',
    'SimpleDistLearner',
    'SimpleLearner',
    'StateShaper',
    'StaticExplorationParameterGenerator',
    'Trainer',
    'TwoPhaseLinearExplorationParameterGenerator',
    'concat_experiences_by_agent',
    'merge_experiences_with_trajectory_boundaries',
    'preprocess',
    'to_device',
    'validate_task_names',
    'merge_experiences_with_trajectory_boundaries'
]
