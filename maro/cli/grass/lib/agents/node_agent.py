# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import multiprocessing
import os
import subprocess
import time

import redis

from .utils import get_node_details, set_node_details

INSPECT_CONTAINER_COMMAND = "docker inspect {container}"
GET_CONTAINERS_COMMAND = "docker container ls -a --format='{{.Names}}'"
UPTIME_COMMAND = "uptime"
FREE_COMMAND = "free"
NVIDIA_SMI_COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"


class NodeAgent:
    def __init__(self, cluster_name: str, node_name: str, master_hostname: str, redis_port: int):
        self._cluster_name = cluster_name
        self._node_name = node_name
        self._master_hostname = master_hostname
        self._redis_port = redis_port

    def start(self) -> None:
        container_tracking_agent = ContainerTrackingAgent(
            cluster_name=self._cluster_name,
            node_name=self._node_name,
            master_hostname=self._master_hostname,
            redis_port=self._redis_port
        )
        container_tracking_agent.start()


class ContainerTrackingAgent(multiprocessing.Process):
    def __init__(
        self, cluster_name: str, node_name: str, master_hostname: str, redis_port: int, check_interval: int = 10
    ):
        super().__init__()
        self._cluster_name = cluster_name
        self._node_name = node_name
        self._redis = redis.Redis(
            host=master_hostname,
            port=redis_port,
            charset="utf-8", decode_responses=True
        )

        self._check_interval = check_interval

    def run(self) -> None:
        while True:
            self._update_node_details()
            time.sleep(self._check_interval)

    def _update_node_details(self) -> None:
        # Get node details
        node_details = get_node_details(
            redis=self._redis,
            cluster_name=self._cluster_name,
            node_name=self._node_name
        )

        # Main update
        self._update_container_details(node_details=node_details)
        self._update_system_resources_details(node_details=node_details)

        # Other update
        node_details['state'] = 'Running'
        node_details['check_time'] = self._redis.time()[0]

        # Save node details
        set_node_details(
            redis=self._redis,
            cluster_name=self._cluster_name,
            node_name=self._node_name,
            node_details=node_details
        )

    def _update_container_details(self, node_details: dict) -> None:
        # Get containers
        completed_process = subprocess.run(
            GET_CONTAINERS_COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
        )
        return_str = completed_process.stdout.strip('\n')
        containers = [] if return_str == '' else return_str.split('\n')

        # Iterate containers
        node_details['containers'] = {}
        container_details = node_details['containers']
        occupied_cpu_sum = 0
        occupied_memory_sum = 0
        occupied_gpu_sum = 0
        for container in containers:
            # Get inspect detail
            completed_process = subprocess.run(
                INSPECT_CONTAINER_COMMAND.format(container=container),
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
            )
            return_str = completed_process.stdout
            inspect_details = json.loads(return_str)[0]

            # Init container details
            container_details[container] = {}

            # Extract occupied resource
            occupied_resource = ContainerTrackingAgent._extract_occupied_resources(inspect_details=inspect_details)
            occupied_cpu_sum += occupied_resource[0]
            occupied_memory_sum += occupied_resource[1]
            occupied_gpu_sum += occupied_resource[2]

            # Extract container state
            container_state = ContainerTrackingAgent._extract_state(
                inspect_details=inspect_details)
            container_details[container]['state'] = container_state

        # Update resources
        node_details['resources']['target_free_cpu'] = node_details['resources']['cpu'] - occupied_cpu_sum
        node_details['resources']['target_free_memory'] = node_details['resources']['memory'] - occupied_memory_sum
        node_details['resources']['target_free_gpu'] = node_details['resources']['gpu'] - occupied_gpu_sum

    def _update_system_resources_details(self, node_details: dict):
        # Get actual cpu
        completed_process = subprocess.run(
            UPTIME_COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
        )
        uptime_str = completed_process.stdout
        split_uptime = uptime_str.split()
        node_details['resources']['actual_free_cpu'] = \
            node_details['resources']['cpu'] - float(split_uptime[-3].replace(',', ''))

        # update actual memory
        completed_process = subprocess.run(
            FREE_COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
        )
        free_str = completed_process.stdout
        split_free = free_str.split()
        node_details['resources']['actual_free_memory'] = float(split_free[12]) / 1024

        # Update actual cpu
        node_details['resources']['actual_free_gpu'] = node_details['resources']['target_free_gpu']
        # Get nvidia-smi result
        try:
            completed_process = subprocess.run(
                NVIDIA_SMI_COMMAND, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
            )
            nvidia_smi_str = completed_process.stdout
            node_details['resources']['actual_gpu_usage'] = f"{float(nvidia_smi_str)}%"
        except Exception:
            pass

    @staticmethod
    def _extract_state(inspect_details: dict) -> dict:
        return inspect_details['State']

    @staticmethod
    def _extract_occupied_resources(inspect_details: dict) -> tuple:
        if inspect_details['State']['Running'] is True:
            occupied_cpu = float(inspect_details['Config']['Labels'].get('required_cpu', 0))
            occupied_memory = float(inspect_details['Config']['Labels'].get('required_memory', '0m').replace("m", ""))
            occupied_gpu = int(inspect_details['Config']['Labels'].get('required_gpu', 0))
            return occupied_cpu, occupied_memory, occupied_gpu
        else:
            return 0, 0, 0


if __name__ == "__main__":
    # FIXME: what about get it from argparse
    with open(os.path.expanduser("~/.maro-local/agents/node_agent.config"), 'r') as fr:
        node_agent_config = json.load(fr)

    node_agent = NodeAgent(
        cluster_name=node_agent_config['cluster_name'],
        node_name=node_agent_config['node_name'],
        master_hostname=node_agent_config['master_hostname'],
        redis_port=node_agent_config['redis_port']
    )
    node_agent.start()
