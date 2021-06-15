# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
import yaml
import random

from maro.rl import (
    DQNConfig, ExperienceManager, OptimOption,
    MultiPhaseLinearExplorationScheduler
)
from maro.simulator import Env
from maro.utils import LogFormat, Logger

from components import VMEnvWrapperForDQN
from agent import QNet, VMDQN, VMExploration
from examples.vm_scheduling.reinforcement_learning.common import (
    VMLearner, RuleAgent, ILPAgent, CombineNet, SequenceNet
)


FILE_PATH = os.path.split(os.path.realpath(__file__))[0]
CONFIG_PATH = os.path.join(FILE_PATH, "config.yml")
with io.open(CONFIG_PATH, "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

LOG_PATH = os.path.join(FILE_PATH, "log", config["experiment_name"])
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
simulation_logger = Logger(tag="simulation", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)
test_simulation_logger = Logger(tag="test_simulation", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)
dqn_logger = Logger(tag="dqn", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)
test_dqn_logger = Logger(tag="test_dqn", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)
ilp_logger = Logger(tag="ilp", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)

MODEL_PATH = os.path.join(FILE_PATH, "log", config["experiment_name"], "models")
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

PICTURE_PATH = os.path.join(FILE_PATH, "log", config["experiment_name"], "pictures")
if not os.path.exists(PICTURE_PATH):
    os.makedirs(PICTURE_PATH)

input_dim = (
    (config["env"]["wrapper"]["pm_num"] * 2)
    * config["env"]["wrapper"]["pm_window_size"]
    + (5 * config["env"]["wrapper"]["vm_window_size"])
)


def get_dqn_policy(agent_config):
    q_net = QNet(
        CombineNet(input_dim=input_dim, **agent_config["model"]["network"]),
        optim_option=OptimOption(**agent_config["model"]["optimization"])
    )
    experience_manager = ExperienceManager(**agent_config["experience_manager"])
    return VMDQN(
        None, q_net, experience_manager, DQNConfig(**agent_config["algorithm_config"])
    )


def get_rule_based_policy(env, agent_config):
    agent = RuleAgent(env, pm_num=agent_config["pm_num"], agent_config=agent_config["algorithm"])
    return agent


def get_ilp_policy(env, agent_config):
    agent = ILPAgent(
        env,
        pm_num=agent_config["pm_num"],
        agent_config=agent_config["algorithm"],
        simulation_logger=simulation_logger,
        ilp_logger=ilp_logger,
        log_path=LOG_PATH
    )
    return agent


if __name__ == "__main__":
    env = Env(**config["env"]["basic"])
    eval_env = Env(**config["eval_env"]["basic"])

    shutil.copy(
        os.path.join(env._business_engine._config_path, "config.yml"),
        os.path.join(LOG_PATH, "BEconfig.yml")
    )
    shutil.copy(CONFIG_PATH, os.path.join(LOG_PATH, "config.yml"))

    if config["seed"] is not None:
        env.set_seed(config["seed"])
        eval_env.set_seed(config["seed"])
        random.seed(config["seed"])

    num_actions = config["policy"]["output_dim"]
    epsilon_greedy = VMExploration(num_actions=num_actions)
    epsilon_greedy.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        **config["exploration"]
    )
    local_learner = VMLearner(
        env=VMEnvWrapperForDQN(env, **config["env"]["wrapper"]),
        policy=get_dqn_policy(config["policy"]),
        auxiliary_policy=get_ilp_policy(env, config["policy"]["ilp_agent"]),
        num_episodes=config["num_episodes"],
        num_steps=config["num_steps"],
        exploration=epsilon_greedy,
        eval_schedule=config["eval_schedule"],
        eval_env=VMEnvWrapperForDQN(eval_env, **config["eval_env"]["wrapper"]),
        auxiliary_prob=config["auxiliary_prob"],
        simulation_logger=simulation_logger,
        eval_simulation_logger=test_simulation_logger,
        rl_logger=dqn_logger,
        eval_rl_logger=test_dqn_logger,
        model_path=MODEL_PATH,
        picture_path=PICTURE_PATH
    )

    local_learner.run()
