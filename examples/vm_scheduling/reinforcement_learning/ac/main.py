# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import yaml
import random
from os import getenv
from os.path import dirname, join, realpath

import numpy as np

from maro.rl import (
    ActorCritic, ActorCriticConfig, ExperienceManager
)
from maro.simulator import Env
from maro.utils import LogFormat, Logger, convert_dottable

from config import agent_config, training_config
from components import VMEnvWrapperForAC
from agent import ACNet, VMActorCritic
from examples.vm_scheduling.reinforcement_learning.common import VMLearner, RuleAgent, ILPAgent


FILE_PATH = os.path.split(os.path.realpath(__file__))[0]
LOG_PATH = os.path.join(FILE_PATH, "log", training_config["experiment_name"])
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
simulation_logger = Logger(tag="simulation", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)
test_simulation_logger = Logger(tag="test_simulation", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)
ac_logger = Logger(tag="ac", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)
test_ac_logger = Logger(tag="test_ac", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)
ilp_logger = Logger(tag="ilp", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)

MODEL_PATH = os.path.join(FILE_PATH, "log", training_config["experiment_name"], "models")
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

PICTURE_PATH = os.path.join(FILE_PATH, "log", training_config["experiment_name"], "pictures")
if not os.path.exists(PICTURE_PATH):
    os.makedirs(PICTURE_PATH)


def get_ac_policy(agent_config):
    ac_net = ACNet(
        component={
            "actor": agent_config["actor_type"](**agent_config["actor"]),
            "critic": agent_config["critic_type"](**agent_config["critic"])
        },
        optim_option={
            "actor":  agent_config["optimization"]["actor"],
            "critic": agent_config["optimization"]["critic"]
        }
    )
    experience_manager = ExperienceManager(**agent_config["experience_manager"])
    return VMActorCritic(
        None, ac_net, experience_manager, ActorCriticConfig(**agent_config["algorithm_config"])
    )


def get_rule_based_policy(env, agent_config):
    agent = RuleAgent(env, pm_num=agent_config["pm_num"], agent_config=agent_config["algorithm"])
    return agent


def get_ilp_policy(env, agent_config):
    agent = ILPAgent(
        env, **agent_config["ilp_agent"],
        simulation_logger=simulation_logger,
        ilp_logger=ilp_logger,
        log_path=LOG_PATH
    )
    return agent


if __name__ == "__main__":
    env = Env(**training_config["env"]["basic"])
    eval_env = Env(**training_config["eval_env"]["basic"])

    if training_config["seed"] is not None:
        env.set_seed(training_config["seed"])
        eval_env.set_seed(training_config["seed"])
        random.seed(training_config["seed"])

    local_learner = VMLearner(
        env=VMEnvWrapperForAC(env, **training_config["env"]["wrapper"]),
        policy=get_ac_policy(agent_config),
        auxiliary_policy=get_rule_based_policy(env, agent_config["rule_agent"]),
        num_episodes=training_config["num_episodes"],
        num_steps=training_config["num_steps"],
        eval_schedule=training_config["eval_schedule"],
        eval_env=VMEnvWrapperForAC(eval_env, **training_config["eval_env"]["wrapper"]),
        auxiliary_prob=training_config["auxiliary_prob"],
        simulation_logger=simulation_logger,
        eval_simulation_logger=test_simulation_logger,
        rl_logger=ac_logger,
        eval_rl_logger=test_ac_logger,
        model_path=MODEL_PATH,
        picture_path=PICTURE_PATH
    )

    local_learner.run()
