# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from policies import agent2policy, policy_dict, policy_update_schedule
from learner import SCLearner
from exploration import exploration_dict, agent2exploration
from env_wrapper import SCEnvWrapper
from config import config
import os
import argparse
import sys
from multiprocessing import Process
from os import getenv, makedirs
from os.path import dirname, join, realpath

from maro.rl import Actor, ActorManager
from maro.simulator import Env
from maro.utils import set_seeds

sc_code_dir = dirname(realpath(__file__))
sys.path.insert(0, sc_code_dir)
# from or_policies import agent2policy, policy_dict, policy_update_schedule

import logging
logging.basicConfig(level=logging.DEBUG)

# for distributed / multi-process training
GROUP = getenv("GROUP", default=config["distributed"]["group"])
REDIS_HOST = config["distributed"]["redis_host"]
REDIS_PORT = config["distributed"]["redis_port"]
NUM_ACTORS = config["distributed"]["num_actors"]

log_dir = join(sc_code_dir, "logs", GROUP)
makedirs(log_dir, exist_ok=True)


def sc_learner():
    # create an actor manager to collect simulation data from multiple actors
    actor_manager = ActorManager(
        NUM_ACTORS, GROUP,
        redis_address=(REDIS_HOST, REDIS_PORT),
        log_enable=False,
        log_dir=log_dir
    )

    # create a learner to start training
    env = SCEnvWrapper(Env(**config["env"]))
    learner = SCLearner(
        policy_dict, agent2policy, config["num_episodes"], policy_update_schedule,
        actor_manager=actor_manager,
        experience_update_interval=config["experience_update_interval"],
        eval_schedule=config["eval_schedule"],
        eval_env=env,
        log_env_metrics=config["log_env_metrics"],
        log_dir=log_dir
    )
    learner.run()


def sc_actor(name: str):
    # create an actor that collects simulation data for the learner.
    actor = Actor(
        SCEnvWrapper(Env(**config["env"])), policy_dict, agent2policy, GROUP,
        exploration_dict=exploration_dict,
        agent2exploration=agent2exploration,
        component_name=name,
        redis_address=(REDIS_HOST, REDIS_PORT),
        log_dir=log_dir
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
        actor_processes = [Process(target=sc_actor, args=(
            f"actor_{i + 1}",)) for i in range(NUM_ACTORS)]
        learner_process = Process(target=sc_learner)

        for i, actor_process in enumerate(actor_processes):
            # this is to ensure that the actors explore differently.
            set_seeds(i)
            actor_process.start()

        learner_process.start()

        for actor_process in actor_processes:
            actor_process.join()

        learner_process.join()
    elif args.whoami == 1:
        sc_learner()
    elif args.whoami == 2:
        sc_actor(getenv("COMPONENT"))
