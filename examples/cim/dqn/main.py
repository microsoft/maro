# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import time
from multiprocessing import Process
from os import makedirs
from os.path import dirname, join, realpath

from maro.rl import (
    BaseActor, DQN, DQNConfig, FullyConnectedBlock, MultiAgentWrapper, SimpleMultiHeadModel,
    TwoPhaseLinearParameterScheduler
)
from maro.simulator import Env
from maro.utils import Logger, set_seeds

from examples.cim.dqn.config import agent_config, training_config
from examples.cim.dqn.training import BasicLearner, BasicRolloutExecutor



def get_dqn_agent():
    q_model = SimpleMultiHeadModel(
        FullyConnectedBlock(**agent_config["model"]), optim_option=agent_config["optimization"]
    )
    return DQN(q_model, DQNConfig(**agent_config["hyper_params"]))


def cim_dqn_learner():
    env = Env(**training_config["env"])
    agent = MultiAgentWrapper({name: get_dqn_agent() for name in env.agent_idx_list})
    scheduler = TwoPhaseLinearParameterScheduler(training_config["max_episode"], **training_config["exploration"])

    log_path = join(dirname(realpath(__file__)), "logs")
    makedirs(log_path, exist_ok=True)
    learner = BasicLearner(
        training_config["group"], training_config["num_actors"], agent, scheduler,
        update_trigger=training_config["learner_update_trigger"],
        logger=Logger(training_config["group"], dump_folder=log_path)
    )

    time.sleep(5)
    learner.run()
    learner.exit()


def cim_dqn_actor():
    env = Env(**training_config["env"])
    agent = MultiAgentWrapper({name: get_dqn_agent() for name in env.agent_idx_list})
    executor = BasicRolloutExecutor(env, agent)
    actor = BaseActor(training_config["group"], executor)
    actor.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--whoami", type=int, choices=[0, 1, 2], default=0,
        help="Identity of this process: 0 - multi-process mode, 1 - learner, 2 - actor"
    )
    
    args = parser.parse_args()
    if args.whoami == 0:
        actor_processes = [Process(target=cim_dqn_actor) for _ in range(training_config["num_actors"])]
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
    else:
        cim_dqn_actor()
