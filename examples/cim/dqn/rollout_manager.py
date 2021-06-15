# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from multiprocessing import Process

from maro.rl import (
    EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler, LocalDecisionGenerator,
    LocalRolloutManager, ParallelRolloutManager, rollout_worker
)
from maro.simulator import Env
from maro.utils import set_seeds

dqn_path = os.path.dirname(os.path.realpath(__file__))  # DQN directory
cim_path = os.path.dirname(dqn_path)  # CIM example directory
sys.path.insert(0, cim_path)
sys.path.insert(0, dqn_path)
from env_wrapper import CIMEnvWrapper
from general import NUM_ACTIONS, config, log_dir
from policy import get_independent_policy


def get_rollout_worker_process():
    env = Env(**config["env"]["basic"])
    decision_generator = LocalDecisionGenerator(
        agent2policy={i: i for i in env.agent_idx_list},
        policies=[get_independent_policy(config["policy"], i, training=False) for i in env.agent_idx_list],
        log_dir=log_dir
    )
    rollout_worker(
        env=CIMEnvWrapper(env, **config["env"]["wrapper"]),
        decision_generator=decision_generator,
        group=config["multi-process"]["group"],
        exploration_dict={f"EpsilonGreedy1": EpsilonGreedyExploration(num_actions=NUM_ACTIONS)},
        agent2exploration={i: f"EpsilonGreedy1" for i in env.agent_idx_list},
        log_dir=log_dir,
        redis_address=(config["multi-process"]["redis_host"], config["multi-process"]["redis_port"])
    )


epsilon_greedy = EpsilonGreedyExploration(num_actions=NUM_ACTIONS)
epsilon_greedy.register_schedule(
    scheduler_cls=MultiPhaseLinearExplorationScheduler,
    param_name="epsilon",
    last_ep=config["num_episodes"],
    **config["exploration"]
)
if config["multi-process"]["rollout_mode"] == "local":
    env = Env(**config["env"]["basic"])
    rollout_manager = LocalRolloutManager(
        env=CIMEnvWrapper(env, **config["env"]["wrapper"]),
        policies=[get_independent_policy(config["policy"], i) for i in env.agent_idx_list],
        agent2policy={i: i for i in env.agent_idx_list},
        exploration_dict={f"EpsilonGreedy1": EpsilonGreedyExploration(num_actions=NUM_ACTIONS)},
        agent2exploration={i: f"EpsilonGreedy1" for i in env.agent_idx_list},
        log_dir=log_dir
    )
else:
    rollout_worker_processes = [
        Process(target=get_rollout_worker_process) for _ in range(config["multi-process"]["num_rollout_workers"])
    ]

    for i, rollout_worker_process in enumerate(rollout_worker_processes):
        set_seeds(i)
        rollout_worker_process.start()

    rollout_manager = ParallelRolloutManager(
        num_rollout_workers=config["multi-process"]["num_rollout_workers"],
        group=config["multi-process"]["group"],
        exploration_dict={f"EpsilonGreedy1": epsilon_greedy},
        # max_receive_attempts=config["multi-process"]["max_receive_attempts"],
        # receive_timeout=config["multi-process"]["receive_timeout"],
        log_dir=log_dir,
        redis_address=(config["multi-process"]["redis_host"], config["multi-process"]["redis_port"])
    )
