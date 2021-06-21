# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler, Learner, LocalLearner
from maro.simulator import Env

async_mode_path = os.path.dirname(os.path.realpath(__file__))  # DQN async mode directory
dqn_path = os.path.dirname(async_mode_path)  # DQN directory
cim_path = os.path.dirname(dqn_path)  # CIM example directory
sys.path.insert(0, cim_path)
sys.path.insert(0, dqn_path)
sys.path.insert(0, async_mode_path)
from env_wrapper import CIMEnvWrapper
from general import NUM_ACTIONS, config, log_dir
from policy import get_independent_policy_for_training
from policy_manager import policy_manager



