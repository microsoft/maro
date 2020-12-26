# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import multiprocessing
import os
import time

import docker
import psutil
import redis

from .utils.details import get_node_details, set_node_details
from .utils.exception import CommandExecutionError
from .utils.resource import BasicResource
from .utils.subprocess import SubProcess

GET_TOTAL_GPU_COUNT_COMMAND = "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"
GET_UTILIZATION_GPUS_COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"


class NodeAgent:
    def __init__(self, cluster_name: str, node_name: str, master_hostname: str, redis_port: int):
        self._cluster_name = cluster_name
        self._node_name = node_name
        self._master_hostname = master_hostname
        self._redis_port = redis_port
        self._redis = redis.Redis(
            host=master_hostname, port=redis_port,
            encoding="utf-8", decode_responses=True
        )

    def start(self) -> None:
        self.init_agent()
        node_tracking_agent = NodeTrackingAgent(
            cluster_name=self._cluster_name, node_name=self._node_name,
            master_hostname=self._master_hostname, redis_port=self._redis_port,
            check_interval=5
        )
        node_tracking_agent.start()

    def init_agent(self) -> None:
        self._init_resources_details()

    def _init_resources_details(self) -> None:
        resource = {}

        # Get cpu info
        resource["cpu"] = psutil.cpu_count()

        # Get memory info
        resource["memory"] = psutil.virtual_memory().total / 1024

        # Get GPU info
        try:
            return_str = SubProcess.run(command=GET_TOTAL_GPU_COUNT_COMMAND)
            resource["gpu"] = int(return_str)
        except CommandExecutionError:
            resource["gpu"] = 0

        # Set resource details
        node_details = get_node_details(
            redis=self._redis,
            cluster_name=self._cluster_name,
            node_name=self._node_name
        )
        node_details["resources"] = resource
        set_node_details(
            redis=self._redis,
            cluster_name=self._cluster_name,
            node_name=self._node_name,
            node_details=node_details
        )


class NodeTrackingAgent(multiprocessing.Process):
    def __init__(
        self,
        cluster_name: str, node_name: str,
        master_hostname: str, redis_port: int,
        check_interval: int = 5
    ):
        super().__init__()
        self._cluster_name = cluster_name
        self._node_name = node_name
        self._redis = redis.Redis(
            host=master_hostname, port=redis_port,
            encoding="utf-8", decode_responses=True
        )
        self._docker = docker.APIClient(base_url="unix:///var/run/docker.sock")

        # Other params.
        self._check_interval = check_interval
        self._container_details = {}

    def run(self) -> None:
        """Start tracking node status and updating details.

        Returns:
            None.
        """
        while True:
            start_time = time.time()
            self._update_details()
            time.sleep(max(self._check_interval - (time.time() - start_time), 0))

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
        node_details["state"]["status"] = "Running"
        node_details["state"]["check_time"] = self._redis.time()[0]

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
        node_details["resources"]["actual_free_cpu"] = node_details["resources"]["cpu"] - psutil.getloadavg()[0]

        # Update actual memory.
        node_details["resources"]["actual_free_memory"] = psutil.virtual_memory().free / 1024

        # Update actual gpu.
        node_details["resources"]["actual_free_gpu"] = node_details["resources"]["target_free_gpu"]
        # Get nvidia-smi result.
        try:
            return_str = SubProcess.run(command=GET_UTILIZATION_GPUS_COMMAND)
            split_str = return_str.split("\n")
            total_usage = 0
            for single_usage in split_str:
                total_usage += float(single_usage)
            node_details["resources"]["actual_gpu_usage"] = f"{float(total_usage) / len(split_str)}%"
        except CommandExecutionError:
            pass

    def _get_inspects_details(self) -> dict:
        """Get inspects_details of containers in the current node.

        Returns:
            dict[str, dict]: container_name to inspect_details mapping.
        """
        # Get container infos in current node.
        container_infos = self._docker.containers(all=True)

        # Build inspect_details and return
        inspects_details = {}
        for container_info in container_infos:
            inspect_details = self._docker.inspect_container(container_info["Id"])
            inspects_details[inspect_details["Config"]["Labels"]["container_name"]] = inspect_details
        return inspects_details

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
    with open(os.path.expanduser("~/.maro-local/agents/maro-node-agent.config"), "r") as fr:
        node_agent_config = json.load(fr)

    node_agent = NodeAgent(
        cluster_name=node_agent_config["cluster_name"],
        node_name=node_agent_config["node_name"],
        master_hostname=node_agent_config["master_hostname"],
        redis_port=node_agent_config["redis_port"]
    )
    node_agent.start()
