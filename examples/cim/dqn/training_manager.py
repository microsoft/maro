# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from multiprocessing import Process

from maro.rl import LocalTrainingManager, ParallelTrainingManager, PolicyServer

dqn_path = os.path.dirname(os.path.realpath(__file__))  # DQN directory
cim_path = os.path.dirname(dqn_path)  # CIM example directory
sys.path.insert(0, cim_path)
sys.path.insert(0, dqn_path)
from general import AGENT_IDS, NUM_POLICY_SERVERS, config, log_dir
from policy import get_independent_policy


def get_policy_server_process(server_id):
    server = PolicyServer(
        policies=[
            get_independent_policy(config["policy"], agent_id)
            for agent_id in AGENT_IDS if agent_id % NUM_POLICY_SERVERS == server_id
        ],
        group=config["multi-process"]["group"],
        name=f"SERVER.{server_id}",
        log_dir=log_dir
    )
    server.run()


policy_list = [get_independent_policy(config["policy"], i) for i in AGENT_IDS]
if config["multi-process"]["policy_training_mode"] == "local":
    training_manager = LocalTrainingManager(policies=policy_list, log_dir=log_dir)
else:
    server_processes = [
        Process(target=get_policy_server_process, args=(server_id,)) for server_id in range(NUM_POLICY_SERVERS)
    ]

    for server_process in server_processes:
        server_process.start() 

    training_manager = ParallelTrainingManager(
        policy2server={id_: f"SERVER.{id_ % NUM_POLICY_SERVERS}" for id_ in AGENT_IDS}, # policy-server mapping
        group=config["multi-process"]["group"],
        log_dir=log_dir
    )
