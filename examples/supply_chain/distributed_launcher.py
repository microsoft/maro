# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import sys
import time
import yaml
from multiprocessing import Process
from os import getenv, makedirs
from os.path import dirname, join, realpath

from maro.rl import (
    Actor, ActorManager, DistLearner, FullyConnectedBlock, LinearParameterScheduler, MultiAgentWrapper,
    OptimOption, SimpleMultiHeadModel
)
from maro.simulator import Env
from maro.utils import set_seeds

sc_code_dir = dirname(realpath(__file__))
sys.path.insert(0, sc_code_dir)
from env_wrapper import SCEnvWrapper
from agent import get_agent_func_map


# Read the configuration
DEFAULT_CONFIG_PATH = join(sc_code_dir, "config.yml")
with open(getenv("CONFIG_PATH", default=DEFAULT_CONFIG_PATH), "r") as config_file:
    config = yaml.safe_load(config_file)

get_producer_agent = get_agent_func_map[config["agent"]["producer"]["algorithm"]]
get_consumer_agent = get_agent_func_map[config["agent"]["consumer"]["algorithm"]]

# Get the env config path
topology = config["training"]["env"]["topology"]
config["training"]["env"]["topology"] = join(sc_code_dir, "envs", topology)

# for distributed / multi-process training
GROUP = getenv("GROUP", default=config["distributed"]["group"])
REDIS_HOST = config["distributed"]["redis_host"]
REDIS_PORT = config["distributed"]["redis_port"]
NUM_ACTORS = config["distributed"]["num_actors"]

log_dir = join(sc_code_dir, "logs", GROUP)
makedirs(log_dir, exist_ok=True)


def sc_learner():
    # create agents that update themselves using experiences collected from the actors.
    env = SCEnvWrapper(Env(**config["training"]["env"]))
    config["agent"]["producer"]["model"]["input_dim"] = config["agent"]["consumer"]["model"]["input_dim"] = env.dim
    producers = {f"producer.{info.id}": get_producer_agent(config["agent"]["producer"]) for info in env.agent_idx_list}
    consumers = {f"consumer.{info.id}": get_consumer_agent(config["agent"]["consumer"]) for info in env.agent_idx_list}
    agent = MultiAgentWrapper({**producers, **consumers})

    # exploration schedule
    scheduler = LinearParameterScheduler(config["training"]["max_episode"], **config["training"]["exploration"])
    
    # create an actor manager to collect simulation data from multiple actors
    actor_manager = ActorManager(
        NUM_ACTORS, GROUP, proxy_options={"redis_address": (REDIS_HOST, REDIS_PORT), "log_enable": False},
        log_dir=log_dir
    )

    # create a learner to start the training process
    learner = DistLearner(
        agent, scheduler, actor_manager,
        agent_update_interval=config["training"]["agent_update_interval"],
        required_actor_finishes=config["distributed"]["required_actor_finishes"],
        discard_stale_experiences=config["distributed"]["discard_stale_experiences"],
        log_dir=log_dir
    )
    learner.run()


def sc_actor():
    # create an env wrapper for roll-out and obtain the input dimension for the agents
    env = SCEnvWrapper(Env(**config["training"]["env"]))
    config["agent"]["producer"]["model"]["input_dim"] = config["agent"]["consumer"]["model"]["input_dim"] = env.dim

    # create agents to interact with the env
    producers = {f"producer.{info.id}": get_producer_agent(config["agent"]["producer"]) for info in env.agent_idx_list}
    consumers = {f"consumer.{info.id}": get_consumer_agent(config["agent"]["consumer"]) for info in env.agent_idx_list}
    agent = MultiAgentWrapper({**producers, **consumers})

    # create an actor that collects simulation data for the learner.
    actor = Actor(env, agent, GROUP, proxy_options={"redis_address": (REDIS_HOST, REDIS_PORT)}, log_dir=log_dir)
    actor.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--whoami", type=int, choices=[0, 1, 2], default=0,
        help="Identity of this process: 0 - multi-process mode, 1 - learner, 2 - actor"
    )

    args = parser.parse_args()
    if args.whoami == 0:
        actor_processes = [Process(target=sc_actor) for i in range(NUM_ACTORS)]
        learner_process = Process(target=sc_learner)

        for i, actor_process in enumerate(actor_processes):
            set_seeds(i)  # this is to ensure that the actors explore differently.
            actor_process.start()

        learner_process.start()

        for actor_process in actor_processes:
            actor_process.join()

        learner_process.join()
    elif args.whoami == 1:
        sc_learner()
    elif args.whoami == 2:
        sc_actor()
