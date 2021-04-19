# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import time
from multiprocessing import Process
from os import makedirs
from os.path import dirname, join, realpath

from maro.rl import BaseActor, DecisionClient
from maro.simulator import Env
from maro.utils import Logger

from examples.cim.ac_gnn.agent import get_gnn_agent, get_experience_pool
from examples.cim.ac_gnn.config import agent_config, training_config
from examples.cim.ac_gnn.shaping import ExperienceShaper, StateShaper
from examples.cim.ac_gnn.training import BasicLearner, BasicRolloutExecutor, learner
from examples.cim.ac_gnn.utils import decision_cnt_analysis, fix_seed, return_scaler


def cim_ac_gnn_actor():
    log_path = join(dirname(realpath(__file__)), "logs")
    makedirs(log_path, exist_ok=True)
    logger = Logger(training_config["group"], dump_folder=log_path)
    # Create a demo environment to retrieve environment information.
    env = Env(**training_config["env"])
    # Add some buffer to prevent overlapping.
    scale_factor, _ = return_scaler(
        env, training_config["env"]["durations"], agent_config["hyper_params"]["reward_discount"]
    )
    logger.info(f"Return values will be scaled down by a factor of {scale_factor}")
    static_nodes = list(env.summary["node_mapping"]["ports"].values())
    dynamic_nodes = list(env.summary["node_mapping"]["vessels"].values())

    state_shaper = StateShaper(
        static_nodes, dynamic_nodes, training_config["env"]["durations"],
        attention_order=agent_config["attention_order"],
        onehot_identity=agent_config["onehot_identity"],
        sequence_buffer_size=agent_config["model"]["sequence_buffer_size"],
        max_value=env.configs["total_containers"]
    )
    state_shaper.compute_static_graph_structure(env)
    experience_shaper = ExperienceShaper(
        static_nodes, dynamic_nodes, training_config["env"]["durations"], state_shaper,
        scale_factor=scale_factor, time_slot=agent_config["hyper_params"]["td_steps"],
        discount_factor=agent_config["hyper_params"]["reward_discount"]
    )

    agent = DecisionClient(
        training_config["group"],
        receive_action_timeout=training_config["actor"]["receive_action_timeout"],
        max_receive_action_attempts=training_config["actor"]["max_receive_action_attempts"],
    )
    executor = BasicRolloutExecutor(
        env, agent, state_shaper, experience_shaper,
        max_null_actions=training_config["actor"]["max_null_actions_per_rollout"], logger=logger
    )
    actor = BaseActor(training_config["group"], executor)
    actor.run()


def cim_ac_gnn_learner():
    log_path = join(dirname(realpath(__file__)), "logs")
    makedirs(log_path, exist_ok=True)
    logger = Logger(training_config["group"], dump_folder=log_path)
    
    # Create a demo environment to retrieve environment information.
    logger.info("Getting experience quantity estimates for each (port, vessel) pair...")
    env = Env(**training_config["env"])
    exp_per_ep = {
        agent_id: cnt * training_config["actor"]["num"] * training_config["train_freq"]
        for agent_id, cnt in decision_cnt_analysis(env, pv=True, buffer_size=8).items()
    }
    logger.info(exp_per_ep)

    static_nodes = list(env.summary["node_mapping"]["ports"].values())
    dynamic_nodes = list(env.summary["node_mapping"]["vessels"].values())
    state_shaper = StateShaper(
        static_nodes, dynamic_nodes, training_config["env"]["durations"],
        attention_order=agent_config["attention_order"],
        onehot_identity=agent_config["onehot_identity"],
        sequence_buffer_size=agent_config["model"]["sequence_buffer_size"],
        max_value=env.configs["total_containers"]
    )
    state_shaper.compute_static_graph_structure(env)
    p_dim, v_dim = state_shaper.get_input_dim("p"), state_shaper.get_input_dim("v")
    exp_pool = get_experience_pool(len(static_nodes), len(dynamic_nodes), exp_per_ep, p_dim, v_dim)
    agent = get_gnn_agent(p_dim, v_dim, state_shaper.p2p_static_graph, exp_pool, logger=logger)
    learner = BasicLearner(
        training_config["group"], training_config["actor"]["num"], training_config["max_episode"], agent,
        update_trigger=training_config["learner"]["update_trigger"],
        inference_trigger=training_config["learner"]["inference_trigger"],
        logger=logger
    )

    time.sleep(5)
    learner.run()
    learner.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--whoami", type=int, choices=[0, 1, 2], default=0,
        help="Identity of this process: 0 - multi-process mode, 1 - learner, 2 - actor"
    )
    
    args = parser.parse_args()
    if args.whoami == 0:
        actor_processes = [Process(target=cim_ac_gnn_actor) for _ in range(training_config["actor"]["num"])]
        # learner_process = Process(target=cim_ac_gnn_learner)

        for actor_process in actor_processes:
            actor_process.start()

        # learner_process.start()

        for actor_process in actor_processes:
            actor_process.join()

        # learner_process.join()
        cim_ac_gnn_learner()
    elif args.whoami == 1:
        cim_ac_gnn_learner()
    else:
        cim_ac_gnn_actor()
