# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.experience.sampler import PrioritizedSampler
from maro.rl.experience import experience_manager
import os
import time
import yaml
from multiprocessing import Process

from maro.rl import (
    Actor, DQN, DQNConfig, EpsilonGreedyExploration, ExperienceManager, FullyConnectedBlock,
    MultiPhaseLinearExplorationScheduler, Learner, LocalLearner, LocalPolicyManager, LocalRolloutManager,
    OptimOption, ParallelPolicyManager, ParallelRolloutManager, PolicyServer,
)
from maro.simulator import Env
from maro.utils import set_seeds

from examples.cim.env_wrapper import CIMEnvWrapper
from examples.cim.dqn.qnet import QNet


FILE_PATH = os.path.dirname(os.path.realpath(__file__))

DEFAULT_CONFIG_PATH = os.path.join(FILE_PATH, "config.yml")
with open(os.getenv("CONFIG_PATH", default=DEFAULT_CONFIG_PATH), "r") as config_file:
    config = yaml.safe_load(config_file)

log_dir = os.path.join(FILE_PATH, "logs", config["experiment_name"])

# agent IDs
AGENT_IDS = Env(**config["env"]["basic"]).agent_idx_list
NUM_POLICY_SERVERS = config["multi-process"]["num_policy_servers"]
POLICY2SERVER = {id_: f"SERVER.{id_ % NUM_POLICY_SERVERS}" for id_ in AGENT_IDS}

# model input and output dimensions
IN_DIM = (
    (config["env"]["wrapper"]["look_back"] + 1)
    * (config["env"]["wrapper"]["max_ports_downstream"] + 1)
    * len(config["env"]["wrapper"]["port_attributes"])
    + len(config["env"]["wrapper"]["vessel_attributes"])
)
OUT_DIM = config["env"]["wrapper"]["num_actions"]


def get_independent_policy(policy_id, training: bool = True):
    cfg = config["policy"]
    qnet = QNet(
        FullyConnectedBlock(input_dim=IN_DIM, output_dim=OUT_DIM, **cfg["model"]["network"]),
        optim_option=OptimOption(**cfg["model"]["optimization"])
    )
    if training:
        experience_manager = ExperienceManager(sampler_cls=PrioritizedSampler, **cfg["experience_manager"]["training"])
    else:
        experience_manager = ExperienceManager(**cfg["experience_manager"]["rollout"])
    return DQN(
        name=policy_id,
        q_net=qnet,
        experience_manager=experience_manager,
        config=DQNConfig(**cfg["algorithm_config"]),
        update_trigger=cfg["update_trigger"],
        warmup=cfg["warmup"]
    )


def local_learner_mode():
    env = Env(**config["env"]["basic"])
    num_actions = config["env"]["wrapper"]["num_actions"]
    epsilon_greedy = EpsilonGreedyExploration(num_actions=num_actions)
    epsilon_greedy.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        last_ep=config["num_episodes"],
        **config["exploration"]
    )
    local_learner = LocalLearner(
        env=CIMEnvWrapper(env, **config["env"]["wrapper"]),
        policies=[get_independent_policy(policy_id=i) for i in env.agent_idx_list],
        agent2policy={i: i for i in env.agent_idx_list},
        num_episodes=config["num_episodes"],
        num_steps=config["num_steps"],
        exploration_dict={f"EpsilonGreedy1": epsilon_greedy},
        agent2exploration={i: f"EpsilonGreedy1" for i in env.agent_idx_list},
        log_dir=log_dir
    )
    local_learner.run()


def get_actor_process():
    env = Env(**config["env"]["basic"])
    num_actions = config["env"]["wrapper"]["num_actions"]
    policy_list = [get_independent_policy(policy_id=i, training=False) for i in env.agent_idx_list]
    actor = Actor(
        env=CIMEnvWrapper(env, **config["env"]["wrapper"]),
        policies=policy_list,
        agent2policy={i: i for i in env.agent_idx_list},
        group=config["multi-process"]["group"],
        exploration_dict={f"EpsilonGreedy1": EpsilonGreedyExploration(num_actions=num_actions)},
        agent2exploration={i: f"EpsilonGreedy1" for i in env.agent_idx_list},
        log_dir=log_dir,
        redis_address=(config["multi-process"]["redis_host"], config["multi-process"]["redis_port"])
    )
    actor.run()


def get_policy_server_process(server_id):
    server = PolicyServer(
        policies=[get_independent_policy(id_) for id_ in AGENT_IDS if id_ % NUM_POLICY_SERVERS == server_id],
        group=config["multi-process"]["group"],
        name=f"SERVER.{server_id}",
        log_dir=log_dir
    )
    server.run()


def get_learner_process():
    epsilon_greedy = EpsilonGreedyExploration(num_actions=config["env"]["wrapper"]["num_actions"])
    epsilon_greedy.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        last_ep=config["num_episodes"],
        **config["exploration"]
    )

    if config["multi-process"]["rollout_mode"] == "local":
        env = Env(**config["env"]["basic"])
        num_actions = config["env"]["wrapper"]["num_actions"]
        rollout_manager = LocalRolloutManager(
            env=CIMEnvWrapper(env, **config["env"]["wrapper"]),
            policies=[get_independent_policy(policy_id=i) for i in env.agent_idx_list],
            agent2policy={i: i for i in env.agent_idx_list},
            exploration_dict={f"EpsilonGreedy1": EpsilonGreedyExploration(num_actions=num_actions)},
            agent2exploration={i: f"EpsilonGreedy1" for i in env.agent_idx_list},
            log_dir=log_dir
        )
    else:
        rollout_manager = ParallelRolloutManager(
            num_actors=config["multi-process"]["num_actors"],
            group=config["multi-process"]["group"],
            exploration_dict={f"EpsilonGreedy1": epsilon_greedy},
            # max_receive_attempts=config["multi-process"]["max_receive_attempts"],
            # receive_timeout=config["multi-process"]["receive_timeout"],
            log_dir=log_dir,
            redis_address=(config["multi-process"]["redis_host"], config["multi-process"]["redis_port"])
        )

    policy_list = [get_independent_policy(policy_id=i) for i in AGENT_IDS]
    if config["multi-process"]["policy_training_mode"] == "local":
        policy_manager = LocalPolicyManager(policies=policy_list, log_dir=log_dir)
    else:
        policy_manager = ParallelPolicyManager(
            policy2server=POLICY2SERVER,
            group=config["multi-process"]["group"],
            log_dir=log_dir
        )
    learner = Learner(
        policy_manager=policy_manager,
        rollout_manager=rollout_manager,
        num_episodes=config["num_episodes"],
        # eval_schedule=config["eval_schedule"],
        log_dir=log_dir
    )

    time.sleep(5)
    learner.run()


def multi_process_mode():
    actor_processes = [Process(target=get_actor_process) for _ in range(config["multi-process"]["num_actors"])]
    if config["multi-process"]["rollout_mode"] == "parallel":
        for i, actor_process in enumerate(actor_processes):
            set_seeds(i)
            actor_process.start()

    if config["multi-process"]["policy_training_mode"] == "parallel":
        server_processes = [
            Process(target=get_policy_server_process, args=(server_id,)) for server_id in range(NUM_POLICY_SERVERS)
        ]
        for server_process in server_processes:
            server_process.start() 

    learner_process = Process(target=get_learner_process)
    learner_process.start()

    if config["multi-process"]["rollout_mode"] == "parallel":
        for actor_process in actor_processes:
            actor_process.join()

    if config["multi-process"]["policy_training_mode"] == "parallel":
        for server_process in server_processes:
            server_process.join()

    learner_process.join()


if __name__ == "__main__":
    if config["mode"] == "local":
        local_learner_mode()
    elif config["mode"] == "multi-process":
        multi_process_mode()
    else:
        print("Two modes are supported: local or multi-process.")
