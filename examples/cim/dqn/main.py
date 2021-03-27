# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from multiprocessing import Process
from os import environ

from maro.rl import (
    Actor, ActorProxy, DQN, DQNConfig, FullyConnectedBlock, MultiAgentWrapper, OffPolicyDistLearner,
    SimpleMultiHeadModel, TwoPhaseLinearParameterScheduler
)
from maro.simulator import Env
from maro.utils import set_seeds

from examples.cim.common import CIMEnvWrapper
from examples.cim.common_config import common_config
from examples.cim.dqn.config import config


GROUP = environ.get("GROUP", config["distributed"]["group"])
REDIS_HOST = environ.get("REDISHOST", config["distributed"]["redis_host"])
REDIS_PORT = environ.get("REDISPORT", config["distributed"]["redis_port"])
NUM_ACTORS = environ.get("NUMACTORS", config["distributed"]["num_actors"])


def get_dqn_agent():
    q_model = SimpleMultiHeadModel(
        FullyConnectedBlock(**config["agent"]["model"]), optim_option=config["agent"]["optimization"]
    )
    return DQN(q_model, DQNConfig(**config["agent"]["hyper_params"]))


def cim_dqn_learner():
    agent = MultiAgentWrapper({name: get_dqn_agent() for name in Env(**config["training"]["env"]).agent_idx_list})
    scheduler = TwoPhaseLinearParameterScheduler(config["training"]["max_episode"], **config["training"]["exploration"])
    actor_proxy = ActorProxy(
        NUM_ACTORS, GROUP,
        proxy_options={"redis_address": (REDIS_HOST, REDIS_PORT)},
        update_trigger=config["distributed"]["learner_update_trigger"],
        **config["training"]["replay_memory"]
    )
    learner = OffPolicyDistLearner(
        actor_proxy, agent, scheduler,
        min_experiences_to_train=config["training"]["min_experiences_to_train"],
        train_iter=config["training"]["train_iter"],
        batch_size=config["training"]["batch_size"]
    )
    learner.run()


def cim_dqn_actor():
    env = Env(**config["training"]["env"])
    agent = MultiAgentWrapper({name: get_dqn_agent() for name in env.agent_idx_list})
    actor = Actor(
        CIMEnvWrapper(env, **common_config), agent, GROUP,
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
