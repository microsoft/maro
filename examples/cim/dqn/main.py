# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import yaml
# from multiprocessing import Process

from maro.rl import (
    DQN, DQNConfig, EpsilonGreedyExploration, ExperienceMemory, FullyConnectedBlock,
    MultiPhaseLinearExplorationScheduler, Learner, LocalLearner, LocalPolicyManager,
    LocalRolloutManager, UniformSampler, OptimOption
)
from maro.simulator import Env
# from maro.utils import set_seeds

from examples.cim.env_wrapper import CIMEnvWrapper
from examples.cim.dqn.qnet import QNet


FILE_PATH = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(FILE_PATH, "log")

DEFAULT_CONFIG_PATH = os.path.join(FILE_PATH, "config.yml")
with open(os.getenv("CONFIG_PATH", default=DEFAULT_CONFIG_PATH), "r") as config_file:
    config = yaml.safe_load(config_file)

# model input and output dimensions
IN_DIM = (
    (config["shaping"]["look_back"] + 1)
    * (config["shaping"]["max_ports_downstream"] + 1)
    * len(config["shaping"]["port_attributes"])
    + len(config["shaping"]["vessel_attributes"])
)
OUT_DIM = config["shaping"]["num_actions"]

# # for distributed / multi-process training
# GROUP = getenv("GROUP", default=config["distributed"]["group"])
# REDIS_HOST = getenv("REDISHOST", default=config["distributed"]["redis_host"])
# REDIS_PORT = getenv("REDISPORT", default=config["distributed"]["redis_port"])
# NUM_ACTORS = int(getenv("NUMACTORS", default=config["distributed"]["num_actors"]))


def get_independent_policy(policy_id):
    cfg = config["policy"]
    qnet = QNet(
        FullyConnectedBlock(input_dim=IN_DIM, output_dim=OUT_DIM, **cfg["model"]),
        optim_option=OptimOption(**cfg["optimization"])
    )
    return DQN(
        name=policy_id,
        q_net=qnet,
        experience_memory=ExperienceMemory(**cfg["experience_memory"]),
        config=DQNConfig(**cfg["algorithm"])
    )


def learner_with_both_local():
    env = Env(**config["training"]["env"])
    policy_list = [get_independent_policy(policy_id=i) for i in env.agent_idx_list]
    agent2policy = {i: i for i in env.agent_idx_list}

    rollout_manager = LocalRolloutManager(
        env=CIMEnvWrapper(env, **config["shaping"]),
        policies=policy_list,
        agent2policy=agent2policy,
        exploration_dict=None,
        agent2exploration=None,
        num_steps=-1,
        eval_env=CIMEnvWrapper(Env(**config["training"]["env"]), **config["shaping"]),
        log_env_metrics=True,
        log_total_reward=True,
        log_dir=log_dir
    )

    policy_manager = LocalPolicyManager(
        policies=policy_list,
        log_dir=log_dir
    )

    learner = Learner(
        policy_manager=policy_manager,
        rollout_manager=rollout_manager,
        num_episodes=config["training"]["num_episodes"],
        eval_schedule=[],
        log_dir=log_dir
    )

    learner.run()


def local_learner_mode():
    env = Env(**config["training"]["env"])

    exploration_config = config["training"]["exploration"]
    exploration_config["splits"] = [(item[0], item[1]) for item in exploration_config["splits"]]
    epsilon_greedy = EpsilonGreedyExploration(num_actions=config["shaping"]["num_actions"])
    epsilon_greedy.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        **exploration_config
    )

    local_learner = LocalLearner(
        env=CIMEnvWrapper(env, **config["shaping"]),
        policies=[get_independent_policy(policy_id=i) for i in env.agent_idx_list],
        agent2policy={i: i for i in env.agent_idx_list},
        num_episodes=config["training"]["num_episodes"],
        num_steps=-1,
        exploration_dict={f"EpsilonGreedy1": epsilon_greedy},
        agent2exploration={i: f"EpsilonGreedy1" for i in env.agent_idx_list},
        eval_schedule=config["training"]["eval_schedule"],
        eval_env=CIMEnvWrapper(Env(**config["training"]["env"]), **config["shaping"]),
        log_env_metrics=True,
        log_total_reward=True,
        log_dir=log_dir
    )

    local_learner.run()


if __name__ == "__main__":
    if config["training"]["env"] == "local":
        local_learner_mode()
    elif config["training"]["env"] == "distributed":
        print("Not implement yet.")
    else:
        print("Two modes are supported: local or distributed.")
