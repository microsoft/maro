# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import sys
import yaml
from multiprocessing import Process
from os import getenv, makedirs
from os.path import dirname, join, realpath

from maro.rl import Actor, ActorManager, Learner, MultiAgentPolicy
from maro.simulator import Env
from maro.utils import set_seeds

sc_code_dir = dirname(realpath(__file__))
sys.path.insert(0, sc_code_dir)
from config import config
from env_wrapper import SCEnvWrapper
from exploration import exploration_dict, agent_to_exploration
from learner import SCLearner
from policies import policy_dict, agent_to_policy


# for distributed / multi-process training
GROUP = getenv("GROUP", default=config["distributed"]["group"])
REDIS_HOST = config["distributed"]["redis_host"]
REDIS_PORT = config["distributed"]["redis_port"]
NUM_ACTORS = config["distributed"]["num_actors"]

log_dir = join(sc_code_dir, "logs", GROUP)
makedirs(log_dir, exist_ok=True)


def get_sc_agents(agent_idx_list, type_):
    assert type_ in {"producer", "consumer"}
    q_model = get_q_model(config["agent"][type_]["model"]) if config["agent"][type_]["share_model"] else None
    alg_type = config["agent"][type_]["algorithm"]
    return {
        f"{type_}.{info.id}": get_agent_func_map[alg_type](config["agent"][type_], q_model=q_model)
        for info in agent_idx_list
    }


def sc_learner():
    # create a multi-agent policy.
    policy = MultiAgentPolicy(
        policy_dict,
        agent_to_policy,
        exploration_dict=exploration_dict,
        agent_to_exploration=agent_to_exploration
    )

    # create an actor manager to collect simulation data from multiple actors
    actor_manager = ActorManager(
        NUM_ACTORS, GROUP, proxy_options={"redis_address": (REDIS_HOST, REDIS_PORT), "log_enable": False},
        log_dir=log_dir
    )

    # create a learner to start training
    learner = SCLearner(
        policy, None, config["num_episodes"],
        actor_manager=actor_manager,
        policy_update_interval=config["policy_update_interval"],
        eval_points=config["eval_points"],
        required_actor_finishes=config["distributed"]["required_actor_finishes"],
        log_env_metrics=config["log_env_metrics"],
        end_of_training_kwargs=config["end_of_training_kwargs"],
        log_dir=log_dir
    )
    learner.run()


def sc_actor(name: str):
    # create an env wrapper for roll-out and obtain the input dimension for the agents
    env = SCEnvWrapper(Env(**config["env"]))
    policy = MultiAgentPolicy(
        policy_dict,
        agent_to_policy,
        exploration_dict=exploration_dict,
        agent_to_exploration=agent_to_exploration
    )
    # create an actor that collects simulation data for the learner.
    actor = Actor(
        env, policy, GROUP,
        proxy_options={"component_name": name, "redis_address": (REDIS_HOST, REDIS_PORT)},
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
        actor_processes = [Process(target=sc_actor, args=(f"actor_{i + 1}",)) for i in range(NUM_ACTORS)]
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
        sc_actor(getenv("COMPONENT"))
