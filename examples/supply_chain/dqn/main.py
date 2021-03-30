# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import yaml
from multiprocessing import Process
from os import getenv
from os.path import dirname, join, realpath

from maro.rl import (
    Actor, ActorProxy, DQN, DQNConfig, FullyConnectedBlock, LinearParameterScheduler, MultiAgentWrapper,
    OffPolicyDistLearner, OptimOption, SimpleMultiHeadModel
)
from maro.simulator import Env
from maro.utils import set_seeds

from examples.supply_chain.env_wrapper import SCEnvWrapper


DEFAULT_CONFIG_PATH = join(dirname(realpath(__file__)), "config.yml")
with open(getenv("CONFIG_PATH", default=DEFAULT_CONFIG_PATH), "r") as config_file:
    config = yaml.safe_load(config_file)

# model input and output dimensions
MANUFACTURER_IN_DIM = 6
MANUFACTURER_OUT_DIM = 10
CONSUMER_IN_DIM = 8
CONSUMER_OUT_DIM = 100

# for distributed / multi-process training
GROUP = getenv("GROUP", default=config["distributed"]["group"])
REDIS_HOST = config["distributed"]["redis_host"]
REDIS_PORT = config["distributed"]["redis_port"]
NUM_ACTORS = int(getenv("NUMACTORS", default=config["distributed"]["num_actors"]))


def get_dqn_agent(in_dim, out_dim):
    q_model = SimpleMultiHeadModel(
        FullyConnectedBlock(input_dim=in_dim, output_dim=out_dim, **config["agent"]["model"]),
        optim_option=OptimOption(**config["agent"]["optimization"])
    )
    return DQN(q_model, DQNConfig(**config["agent"]["hyper_params"]))


def get_sc_agents(agent_ids):
    manufacturer_agents = {
        id_: get_dqn_agent(MANUFACTURER_IN_DIM, MANUFACTURER_OUT_DIM)
        for type_, id_ in agent_ids if type_ == "manufacture"
    }
    consumer_agents = {
        id_: get_dqn_agent(CONSUMER_IN_DIM, CONSUMER_OUT_DIM)
        for type_, id_ in agent_ids if type_ == "consumer"
    }
    return MultiAgentWrapper({**manufacturer_agents, **consumer_agents})


def sc_dqn_learner():
    agent = get_sc_agents(Env(**config["training"]["env"]).agent_idx_list)
    scheduler = LinearParameterScheduler(config["training"]["max_episode"], **config["training"]["exploration"])
    actor_proxy = ActorProxy(
        NUM_ACTORS, GROUP,
        proxy_options={"redis_address": (REDIS_HOST, REDIS_PORT)},
        update_trigger=config["distributed"]["learner_update_trigger"],
        replay_memory_size=config["training"]["replay_memory"]["size"],
        replay_memory_overwrite_type=config["training"]["replay_memory"]["overwrite_type"]
    )
    learner = OffPolicyDistLearner(
        actor_proxy, agent, scheduler,
        min_experiences_to_train=config["training"]["min_experiences_to_train"],
        train_iter=config["training"]["train_iter"],
        batch_size=config["training"]["batch_size"]
    )
    learner.run()


def sc_dqn_actor():
    env = Env(**config["training"]["env"])
    agent = get_sc_agents(env.agent_idx_list)
    actor = Actor(
        SCEnvWrapper(env), agent, GROUP,
        replay_sync_interval=config["distributed"]["replay_sync_interval"],
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
        actor_processes = [Process(target=sc_dqn_actor) for i in range(NUM_ACTORS)]
        learner_process = Process(target=sc_dqn_learner)

        for i, actor_process in enumerate(actor_processes):
            set_seeds(i)  # this is to ensure that the actors explore differently.
            actor_process.start()

        learner_process.start()

        for actor_process in actor_processes:
            actor_process.join()

        learner_process.join()
    elif args.whoami == 1:
        sc_dqn_learner()
    elif args.whoami == 2:
        sc_dqn_actor()
