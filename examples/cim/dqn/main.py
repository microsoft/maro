# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import yaml
from multiprocessing import Process

from maro.rl import (
    Actor, DQN, DQNConfig, EpsilonGreedyExploration, ExperienceMemory, FullyConnectedBlock,
    MultiPhaseLinearExplorationScheduler, Learner, LocalLearner, LocalPolicyManager,
    LocalRolloutManager, OptimOption, ParallelRolloutManager, UniformSampler
)
from maro.simulator import Env
from maro.utils import set_seeds

from examples.cim.env_wrapper import CIMEnvWrapper
from examples.cim.dqn.qnet import QNet


FILE_PATH = os.path.dirname(os.path.realpath(__file__))

DEFAULT_CONFIG_PATH = os.path.join(FILE_PATH, "config.yml")
with open(os.getenv("CONFIG_PATH", default=DEFAULT_CONFIG_PATH), "r") as config_file:
    config = yaml.safe_load(config_file)

log_dir = os.path.join(FILE_PATH, "log", config["experiment_name"])

# model input and output dimensions
IN_DIM = (
    (config["shaping"]["look_back"] + 1)
    * (config["shaping"]["max_ports_downstream"] + 1)
    * len(config["shaping"]["port_attributes"])
    + len(config["shaping"]["vessel_attributes"])
)
OUT_DIM = config["shaping"]["num_actions"]


def get_independent_policy(policy_id):
    cfg = config["policy"]
    qnet = QNet(
        FullyConnectedBlock(input_dim=IN_DIM, output_dim=OUT_DIM, **cfg["model"]),
        optim_option=OptimOption(**cfg["optimization"])
    )
    return DQN(
        name=policy_id,
        q_net=qnet,
        experience_memory=ExperienceMemory(**cfg["experience_memory"]),
        config=DQNConfig(**cfg["algorithm"])
    )


def local_learner_mode():
    env = Env(**config["training"]["env"])

    exploration_config = config["training"]["exploration"]
    exploration_config["splits"] = [(item[0], item[1]) for item in exploration_config["splits"]]
    epsilon_greedy = EpsilonGreedyExploration(num_actions=config["shaping"]["num_actions"])
    epsilon_greedy.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        **exploration_config
    )

    local_learner = LocalLearner(
        env=CIMEnvWrapper(env, **config["shaping"]),
        policies=[get_independent_policy(policy_id=i) for i in env.agent_idx_list],
        agent2policy={i: i for i in env.agent_idx_list},
        num_episodes=config["training"]["num_episodes"],
        num_steps=-1,
        exploration_dict={f"EpsilonGreedy1": epsilon_greedy},
        agent2exploration={i: f"EpsilonGreedy1" for i in env.agent_idx_list},
        eval_schedule=config["training"]["eval_schedule"],
        eval_env=CIMEnvWrapper(Env(**config["training"]["env"]), **config["shaping"]),
        log_env_metrics=True,
        log_total_reward=True,
        log_dir=log_dir
    )

    local_learner.run()


def get_dqn_actor_process():
    env = Env(**config["training"]["env"])
    policy_list = [get_independent_policy(policy_id=i) for i in env.agent_idx_list]

    actor = Actor(
        env=CIMEnvWrapper(env, **config["shaping"]),
        policies=policy_list,
        agent2policy={i: i for i in env.agent_idx_list},
        group=config["training"]["multi-process"]["group"],
        exploration_dict=None,
        agent2exploration=None,
        eval_env=CIMEnvWrapper(Env(**config["training"]["env"]), **config["shaping"]),
        log_dir=log_dir,
        redis_address=(
            config["training"]["multi-process"]["redis_host"],
            config["training"]["multi-process"]["redis_port"]
        )
    )
    actor.run()


def get_dqn_learner_process():
    env = Env(**config["training"]["env"])
    policy_list = [get_independent_policy(policy_id=i) for i in env.agent_idx_list]

    policy_manager = LocalPolicyManager(policies=policy_list, log_dir=log_dir)

    rollout_manager = ParallelRolloutManager(
        num_actors=config["training"]["multi-process"]["num_actors"],
        group=config["training"]["multi-process"]["group"],
        num_steps=-1,
        required_finishes=None,
        max_staleness=0,
        num_eval_actors=config["training"]["multi-process"]["num_eval_actors"],
        log_env_metrics=False,
        log_dir=log_dir,
        redis_address=(
            config["training"]["multi-process"]["redis_host"],
            config["training"]["multi-process"]["redis_port"]
        )
    )

    learner = Learner(
        policy_manager=policy_manager,
        rollout_manager=rollout_manager,
        num_episodes=config["training"]["num_episodes"],
        eval_schedule=[],
        log_dir=log_dir
    )
    learner.run()


def multi_process_mode():
    actor_processes = [Process(target=get_dqn_actor_process) for i in range(config["training"]["multi-process"]["num_actors"])]
    for i, actor_process in enumerate(actor_processes):
        set_seeds(i)
        actor_process.start()

    learner_process = Process(target=get_dqn_learner_process)
    learner_process.start()

    for actor_process in actor_processes:
        actor_process.join()
    learner_process.join()


if __name__ == "__main__":
    if config["training"]["mode"] == "local":
        local_learner_mode()
    elif config["training"]["mode"] == "multi-process":
        multi_process_mode()
    else:
        print("Two modes are supported: local or multi-process.")

