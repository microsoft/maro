# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import sys
from argparse import ArgumentParser
from multiprocessing import Process

import yaml

from maro.rl import Actor, Learner, LocalLearner, LocalPolicyManager, ParallelRolloutManager
from maro.simulator import Env
from maro.simulator.scenarios.supply_chain.world import AgentInfo
from maro.utils import set_seeds

from examples.supply_chain.env_wrapper import SCEnvWrapper
from examples.supply_chain.exploration import get_exploration_mapping
# from examples.supply_chain.learner import SCLearner
from examples.supply_chain.policies import get_policy_mapping, get_replay_agent_ids
from examples.supply_chain.render_tools import SimulationTracker


# logging.basicConfig(level=logging.DEBUG)

SC_CODE_DIR = os.path.dirname(os.path.realpath(__file__))
config_path = os.getenv("CONFIG_PATH", default=os.path.join(SC_CODE_DIR, "config.yml"))
with open(config_path, "r") as config_file:
    CONFIG = yaml.safe_load(config_file)
LOG_DIR = os.path.join(SC_CODE_DIR, "logs", CONFIG["experiment_name"])
OUTPUT_DIR = os.path.join(LOG_DIR, "output")

# AgentInfo = namedtuple("AgentInfo", ("id", "agent_type", "is_facility", "sku", "facility_id", "parent_id"))
def agent_info_to_agent_id(info: AgentInfo) -> str:
    return f"{info.agent_type}.{info.id}"

def single_thread_mode(config, env_wrapper):
    exploration_dict, agent2exploration = get_exploration_mapping(config)
    policies, agent2policy = get_policy_mapping(config)

    # create a learner to start training
    learner = LocalLearner(
        env=env_wrapper,
        policies=policies,
        agent2policy=agent2policy,
        num_episodes=config["num_episodes"],
        num_steps=-1,
        exploration_dict=exploration_dict,
        agent2exploration=agent2exploration,
        eval_schedule=config["eval_schedule"],
        eval_env=None,
        early_stopper=None,
        log_env_summary=config["log_env_summary"],
        log_dir=LOG_DIR
    )

    tracker = SimulationTracker(60, 1, env_wrapper, learner)
    tracker.run_and_render(
        loc_path=OUTPUT_DIR,
        facility_types=["productstore"]
    )

def sc_learner(config):
    policies, _ = get_policy_mapping(config)

    policy_manager = LocalPolicyManager(policies=policies, log_dir=LOG_DIR)

    rollout_manager = ParallelRolloutManager(
        num_actors=config["distributed"]["num_actors"],
        group=config["distributed"]["group"],
        num_steps=-1,
        max_receive_attempts=None,
        receive_timeout=None,
        max_staleness=0,
        num_eval_actors=1,
        log_env_summary=True,
        log_dir=LOG_DIR,
        component_name="rollout_manager",
        redis_address=(config["distributed"]["redis_host"], config["distributed"]["redis_port"]),
    )

    learner = Learner(
        policy_manager=policy_manager,
        rollout_manager=rollout_manager,
        num_episodes=config["num_episodes"],
        eval_schedule=config["eval_schedule"],
        early_stopper=None,
        log_dir=LOG_DIR
    )
    learner.run()

def sc_actor(component_name: str, config, env_wrapper):
    policies, agent2policy = get_policy_mapping(config)
    exploration_dict, agent2exploration = get_exploration_mapping(config)
    actor = Actor(
        env=env_wrapper,
        policies=policies,
        agent2policy=agent2policy,
        group=config["distributed"]["group"],
        exploration_dict=exploration_dict,
        agent2exploration=agent2exploration,
        eval_env=SCEnvWrapper(Env(**config["env"]), replay_agent_ids=config["replay_agent_ids"]),
        log_dir=LOG_DIR,
        component_name=component_name,
        redis_address=(config["distributed"]["redis_host"], config["distributed"]["redis_port"]),
    )
    actor.run()

def multi_process_mode(config):
    actor_processes = [
        Process(
            target=sc_actor,
            args=(f"actor_{i + 1}", config, SCEnvWrapper(Env(**config["env"]), replay_agent_ids=config["replay_agent_ids"]), )
        )
        for i in range(config["distributed"]["num_actors"])
    ]
    learner_process = Process(target=sc_learner, args=(config, ))

    for i, actor_process in enumerate(actor_processes):
        set_seeds(i)
        actor_process.start()
    learner_process.start()

    for actor_process in actor_processes:
        actor_process.join()
    learner_process.join()


if __name__ == "__main__":
    CONFIG["env"]["topology"] = os.path.join(SC_CODE_DIR, "topologies", CONFIG["env"]["topology"])

    env = Env(**CONFIG["env"])
    CONFIG["agent_id_list"] = [agent_info_to_agent_id(info) for info in env.agent_idx_list]
    CONFIG["replay_agent_ids"] = get_replay_agent_ids(CONFIG["agent_id_list"])
    env_wrapper = SCEnvWrapper(env, replay_agent_ids=CONFIG["replay_agent_ids"])

    CONFIG["policy"]["consumer"]["model"]["network"]["input_dim"] = env_wrapper.dim
    CONFIG["policy"]["producer"]["model"]["network"]["input_dim"] = env_wrapper.dim
    CONFIG["policy"]["consumerstore"]["model"]["network"]["input_dim"] = env_wrapper.dim

    parser = ArgumentParser()
    parser.add_argument(
        "-w", "--whoami", type=int, choices=[-1, 0, 1, 2], default=0,
        help="Identify of this process: -1 - single thread mode, 0 - multi-process mode, 1 - learner, 2 - actor"
    )
    args = parser.parse_args()

    if args.whoami == -1:
        single_thread_mode(CONFIG, env_wrapper)
    elif args.whoami == 0:
        if CONFIG["distributed"]["redis_host"] != "localhost":
            CONFIG["distributed"]["redis_host"] = "localhost"
            print(f"******** [multi-process mode] automatically change the 'redis_host' field to 'localhost' ********")
        multi_process_mode(CONFIG)
    elif args.whoami == 1:
        sc_learner(CONFIG)
    elif args.whoami == 2:
        component_name = os.getenv("COMPONENT")
        sc_actor(component_name, CONFIG, env_wrapper)
