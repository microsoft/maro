# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import multiprocessing
import os
import subprocess
import time

import redis

from .resource import BasicResource
from .utils import get_node_details, set_node_details

INSPECT_CONTAINER_COMMAND = "docker inspect {containers}"
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
        container_tracking_agent = NodeTrackingAgent(
            cluster_name=self._cluster_name, node_name=self._node_name,
            master_hostname=self._master_hostname, redis_port=self._redis_port
        )
        container_tracking_agent.start()


class NodeTrackingAgent(multiprocessing.Process):
    def __init__(
        self, cluster_name: str, node_name: str, master_hostname: str, redis_port: int,
        check_interval: int = 10
    ):
        super().__init__()
        self._cluster_name = cluster_name
        self._node_name = node_name
        self._redis = redis.Redis(
            host=master_hostname,
            port=redis_port,
            charset="utf-8", decode_responses=True
        )

        # Other params.
        self._check_interval = check_interval
        self._container_details = {}

    def run(self) -> None:
        """Start tracking node status and updating details.

        Returns:
            None.
        """
        while True:
            self._update_details()
            time.sleep(self._check_interval)

    def _update_details(self) -> None:
        """Update details.

        Returns:
            None.
        """
        # Get or init details.
        node_details = get_node_details(
            redis=self._redis,
            cluster_name=self._cluster_name,
            node_name=self._node_name
        )
        node_details["containers"] = {}
        containers_details = node_details["containers"]
        inspects_details = self._get_inspects_details()

        # Major updates.
        self._update_containers_details(inspects_details=inspects_details, containers_details=containers_details)
        self._update_occupied_resources(inspects_details=inspects_details, node_details=node_details)
        self._update_actual_resources(node_details=node_details)

        # Other updates.
        node_details["state"] = "Running"
        node_details["check_time"] = self._redis.time()[0]

        # Save details.
        set_node_details(
            redis=self._redis,
            cluster_name=self._cluster_name,
            node_name=self._node_name,
            node_details=node_details
        )

    @staticmethod
    def _update_containers_details(inspects_details: dict, containers_details: dict) -> None:
        """Update containers_details from inspects_details.

        Args:
            inspects_details: Details of container inspections.
            containers_details: Details of containers in the current node.

        Returns:
            None.
        """
        # Iterate inspects_details.
        for container_name, inspect_details in inspects_details.items():
            # Extract container state and labels.
            containers_details[container_name] = NodeTrackingAgent._extract_labels(inspect_details=inspect_details)
            containers_details[container_name]["state"] = NodeTrackingAgent._extract_state(
                inspect_details=inspect_details)

    @staticmethod
    def _update_occupied_resources(inspects_details: dict, node_details: dict) -> None:
        """Update occupied resources from containers' inspects_details.

        Args:
            inspects_details: Details of container inspections.
            node_details: Details of the current node.

        Returns:
            None.
        """
        # Init params.
        occupied_cpu_sum = 0.0
        occupied_memory_sum = 0.0
        occupied_gpu_sum = 0.0

        # Iterate inspects_details.
        for _, inspect_details in inspects_details.items():
            # Extract occupied resource.
            occupied_resource = NodeTrackingAgent._extract_occupied_resources(inspect_details=inspect_details)
            occupied_cpu_sum += occupied_resource.cpu
            occupied_memory_sum += occupied_resource.memory
            occupied_gpu_sum += occupied_resource.gpu

        # Update target resources.
        node_details["resources"]["target_free_cpu"] = node_details["resources"]["cpu"] - occupied_cpu_sum
        node_details["resources"]["target_free_memory"] = node_details["resources"]["memory"] - occupied_memory_sum
        node_details["resources"]["target_free_gpu"] = node_details["resources"]["gpu"] - occupied_gpu_sum

    @staticmethod
    def _update_actual_resources(node_details: dict) -> None:
        """Update actual resources status from operating system.

        Args:
            node_details: Details of the current node.

        Returns:
            None.
        """
        # Update actual cpu.
        completed_process = subprocess.run(
            UPTIME_COMMAND,
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
        )
        uptime_str = completed_process.stdout
        split_uptime = uptime_str.split()
        node_details["resources"]["actual_free_cpu"] = (
            node_details["resources"]["cpu"]
            - float(split_uptime[-3].replace(",", ""))
        )

        # Update actual memory.
        completed_process = subprocess.run(
            FREE_COMMAND,
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
        )
        free_str = completed_process.stdout
        split_free = free_str.split()
        node_details["resources"]["actual_free_memory"] = float(split_free[12]) / 1024

        # Update actual gpu.
        node_details["resources"]["actual_free_gpu"] = node_details["resources"]["target_free_gpu"]
        # Get nvidia-smi result.
        try:
            completed_process = subprocess.run(
                NVIDIA_SMI_COMMAND,
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
            )
            nvidia_smi_str = completed_process.stdout
            node_details["resources"]["actual_gpu_usage"] = f"{float(nvidia_smi_str)}%"
        except Exception:
            pass

    @staticmethod
    def _get_inspects_details() -> dict:
        """Get inspects_details of containers in the current node.

        Returns:
            dict[str, dict]: container_name to inspect_details mapping.
        """
        # Get containers in current node.
        completed_process = subprocess.run(
            GET_CONTAINERS_COMMAND,
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
        )
        return_str = completed_process.stdout.strip("\n")
        containers = [] if return_str == "" else return_str.split("\n")
        if len(containers) == 0:
            return {}

        # Get inspect_details_list then build inspects_details.
        completed_process = subprocess.run(
            INSPECT_CONTAINER_COMMAND.format(containers=" ".join(containers)),
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
        )
        return_str = completed_process.stdout
        inspect_details_list = json.loads(return_str)
        return {inspect_details["Config"]["Labels"]["container_name"]: inspect_details
                for inspect_details in inspect_details_list}

    @staticmethod
    def _extract_state(inspect_details: dict) -> dict:
        return inspect_details["State"]

    @staticmethod
    def _extract_labels(inspect_details: dict) -> dict:
        return inspect_details["Config"]["Labels"]

    @staticmethod
    def _extract_occupied_resources(inspect_details: dict) -> BasicResource:
        """Extract occupied resources info from inspect_details.

        Args:
            inspect_details: Details of the container inspection.

        Returns:
            BasicResource: Occupied resource of the current container.
        """
        if inspect_details["State"]["Running"] is True:
            occupied_cpu = float(inspect_details["Config"]["Labels"].get("cpu", 0))
            occupied_memory = float(inspect_details["Config"]["Labels"].get("memory", "0m").replace("m", ""))
            occupied_gpu = int(inspect_details["Config"]["Labels"].get("gpu", 0))
            return BasicResource(cpu=occupied_cpu, memory=occupied_memory, gpu=occupied_gpu)
        else:
            return BasicResource(cpu=0, memory=0, gpu=0)


if __name__ == "__main__":
    # FIXME: what about get it from argparse?
    with open(os.path.expanduser("~/.maro-local/agents/node_agent.config"), "r") as fr:
        node_agent_config = json.load(fr)

    node_agent = NodeAgent(
        cluster_name=node_agent_config["cluster_name"],
        node_name=node_agent_config["node_name"],
        master_hostname=node_agent_config["master_hostname"],
        redis_port=node_agent_config["redis_port"]
    )
    node_agent.start()
