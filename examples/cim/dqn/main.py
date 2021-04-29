# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import yaml
from multiprocessing import Process
from os import getenv
from os.path import dirname, join, realpath

from maro.rl import (
    Actor, ActorManager, DQN, DQNConfig, DistLearner, FullyConnectedBlock, MultiAgentWrapper, OptimOption,
    SimpleMultiHeadModel, TwoPhaseLinearParameterScheduler
)
from maro.simulator import Env
from maro.utils import set_seeds

from examples.cim.env_wrapper import CIMEnvWrapper


DEFAULT_CONFIG_PATH = join(dirname(realpath(__file__)), "config.yml")
with open(getenv("CONFIG_PATH", default=DEFAULT_CONFIG_PATH), "r") as config_file:
    config = yaml.safe_load(config_file)

# model input and output dimensions
IN_DIM = (
    (config["shaping"]["look_back"] + 1) *
    (config["shaping"]["max_ports_downstream"] + 1) *
    len(config["shaping"]["port_attributes"]) +
    len(config["shaping"]["vessel_attributes"])
)
OUT_DIM = config["shaping"]["num_actions"]

# for distributed / multi-process training
GROUP = getenv("GROUP", default=config["distributed"]["group"])
REDIS_HOST = getenv("REDISHOST", default=config["distributed"]["redis_host"])
REDIS_PORT = getenv("REDISPORT", default=config["distributed"]["redis_port"])
NUM_ACTORS = int(getenv("NUMACTORS", default=config["distributed"]["num_actors"]))


def get_dqn_agent():
    cfg = config["agent"]
    q_model = SimpleMultiHeadModel(
        FullyConnectedBlock(input_dim=IN_DIM, output_dim=OUT_DIM, **cfg["model"]),
        optim_option=OptimOption(**cfg["optimization"])
    )
    return DQN(q_model, DQNConfig(**cfg["algorithm"]), **cfg["experience_memory"])


def cim_dqn_learner():
    agent = MultiAgentWrapper({name: get_dqn_agent() for name in Env(**config["training"]["env"]).agent_idx_list})
    scheduler = TwoPhaseLinearParameterScheduler(config["training"]["max_episode"], **config["training"]["exploration"])
    actor_manager = ActorManager(
        NUM_ACTORS, GROUP, proxy_options={"redis_address": (REDIS_HOST, REDIS_PORT), "log_enable": False}
    )
    learner = DistLearner(
        agent, scheduler, actor_manager,
        agent_update_interval=config["training"]["agent_update_interval"],
        required_actor_finishes=config["distributed"]["required_actor_finishes"],
        discard_stale_experiences=False
    )
    learner.run()


def cim_dqn_actor():
    env = Env(**config["training"]["env"])
    agent = MultiAgentWrapper({name: get_dqn_agent() for name in env.agent_idx_list})
    actor = Actor(
        CIMEnvWrapper(env, **config["shaping"]), agent, GROUP,
        proxy_options={"redis_address": (REDIS_HOST, REDIS_PORT)}
    )
    actor.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--whoami", type=int, choices=[0, 1, 2], default=0,
        help="Identity of this process: 0 - multi-process mode, 1 - learner, 2 - actor"
    )

    args = parser.parse_args()
    if args.whoami == 0:
        actor_processes = [Process(target=cim_dqn_actor) for i in range(NUM_ACTORS)]
        learner_process = Process(target=cim_dqn_learner)

        for i, actor_process in enumerate(actor_processes):
            set_seeds(i)  # this is to ensure that the actors explore differently.
            actor_process.start()

        learner_process.start()

        for actor_process in actor_processes:
            actor_process.join()

        learner_process.join()
    elif args.whoami == 1:
        cim_dqn_learner()
    elif args.whoami == 2:
        cim_dqn_actor()
