# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import yaml
# from multiprocessing import Process

from maro.rl import (
    DQN, DQNConfig, EpsilonGreedyExploration, ExperienceMemory, FullyConnectedBlock,
    MultiPhaseLinearExplorationScheduler, Learner, LocalLearner, LocalPolicyManager,
    LocalRolloutManager, OptimOption
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
    (config["env"]["wrapper"]["look_back"] + 1)
    * (config["env"]["wrapper"]["max_ports_downstream"] + 1)
    * len(config["env"]["wrapper"]["port_attributes"])
    + len(config["env"]["wrapper"]["vessel_attributes"])
)
OUT_DIM = config["env"]["wrapper"]["num_actions"]

# # for distributed / multi-process training
# GROUP = getenv("GROUP", default=config["distributed"]["group"])
# REDIS_HOST = getenv("REDISHOST", default=config["distributed"]["redis_host"])
# REDIS_PORT = getenv("REDISPORT", default=config["distributed"]["redis_port"])
# NUM_ACTORS = int(getenv("NUMACTORS", default=config["distributed"]["num_actors"]))


def get_independent_policy(policy_id):
    cfg = config["policy"]
    qnet = QNet(
        FullyConnectedBlock(input_dim=IN_DIM, output_dim=OUT_DIM, **cfg["model"]["network"]),
        optim_option=OptimOption(**cfg["model"]["optimization"])
    )
    return DQN(
        name=policy_id,
        q_net=qnet,
        experience_memory=ExperienceMemory(**cfg["experience_memory"]),
        config=DQNConfig(**cfg["algorithm_config"])
    )


def local_learner_mode():
    env = Env(**config["env"]["basic"])
    num_actions = config["env"]["wrapper"]["num_actions"]
    epsilon_greedy = EpsilonGreedyExploration(num_actions=num_actions)
    epsilon_greedy.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        **config["exploration"]
    )
    local_learner = LocalLearner(
        env=CIMEnvWrapper(env, **config["env"]["wrapper"]),
        policies=[get_independent_policy(policy_id=i) for i in env.agent_idx_list],
        agent2policy={i: i for i in env.agent_idx_list},
        num_episodes=config["num_episodes"],
        num_steps=config["num_steps"],
        exploration_dict={f"EpsilonGreedy1": epsilon_greedy},
        agent2exploration={i: f"EpsilonGreedy1" for i in env.agent_idx_list},
        eval_schedule=config["eval_schedule"],
        log_env_summary=True,
        log_dir=log_dir
    )

    local_learner.run()


if __name__ == "__main__":
    if config["mode"] == "local":
        local_learner_mode()
    elif config["mode"] == "distributed":
        print("Not implement yet.")
    else:
        print("Two modes are supported: local or distributed.")
