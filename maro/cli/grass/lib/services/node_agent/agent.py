# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import logging
import os
import signal
import sys
import threading
import time
from multiprocessing.pool import ThreadPool

import psutil
import redis

from ..utils.docker_controller import DockerController
from ..utils.exception import CommandExecutionError
from ..utils.params import NodeStatus
from ..utils.redis_controller import RedisController
from ..utils.resource import BasicResource
from ..utils.subprocess import SubProcess

GET_TOTAL_GPU_COUNT_COMMAND = "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"
GET_UTILIZATION_GPUS_COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"

NODE_DETAILS_LOCK = threading.Lock()


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
        self._redis_controller = RedisController(redis=self._redis)

        # Init agents.
        self.load_image_agent = LoadImageAgent(
            cluster_name=self._cluster_name, node_name=self._node_name,
            master_hostname=self._master_hostname, redis_port=self._redis_port
        )
        self.node_tracking_agent = NodeTrackingAgent(
            cluster_name=self._cluster_name, node_name=self._node_name,
            master_hostname=self._master_hostname, redis_port=self._redis_port
        )

        # When SIGTERM, gracefully exit.
        signal.signal(signal.SIGTERM, self.gracefully_exit)

    def start(self) -> None:
        self.init_agent()

        # Start agents.
        self.node_tracking_agent.start()
        self.load_image_agent.start()

        # Wait joins.
        self.node_tracking_agent.join()
        self.load_image_agent.join()

    def gracefully_exit(self, signum, frame) -> None:
        """ Gracefully exit when SIGTERM.

        If we get SIGKILL here, it means that the node is not stopped properly,
        the status of the node remains 'RUNNING'.
        Then MARO Master will scan the status of all nodes, and label it 'anomaly'
        because of the incorrect check_time and node status.

        Returns:
            None.
        """
        # Stop agents
        self.node_tracking_agent.stop()
        self.load_image_agent.stop()

        # Set STOPPED state
        node_details = self._redis_controller.get_node_details(
            cluster_name=self._cluster_name,
            node_name=self._node_name
        )
        node_details["state"]["status"] = NodeStatus.STOPPED
        node_details["state"]["check_time"] = self._redis.time()[0]
        self._redis_controller.set_node_details(
            cluster_name=self._cluster_name,
            node_name=self._node_name,
            node_details=node_details
        )
        sys.exit(0)

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
        node_details = self._redis_controller.get_node_details(
            cluster_name=self._cluster_name,
            node_name=self._node_name
        )
        node_details["resources"] = resource
        self._redis_controller.set_node_details(
            cluster_name=self._cluster_name,
            node_name=self._node_name,
            node_details=node_details
        )


class NodeTrackingAgent(threading.Thread):
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
        self._redis_controller = RedisController(redis=self._redis)

        # Other params.
        self._check_interval = check_interval
        self._container_details = {}
        self._is_terminated = False

    def run(self) -> None:
        """Start tracking node status and updating details.

        Returns:
            None.
        """
        while not self._is_terminated:
            start_time = time.time()
            self._update_details()
            time.sleep(max(self._check_interval - (time.time() - start_time), 0))

    def stop(self):
        self._is_terminated = True

    def _update_details(self) -> None:
        """Update details.

        Returns:
            None.
        """
        # Get or init details.

        with NODE_DETAILS_LOCK:
            node_details = self._redis_controller.get_node_details(
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
            node_details["state"]["status"] = NodeStatus.RUNNING
            node_details["state"]["check_time"] = self._redis.time()[0]

            # Save details.
            self._redis_controller.set_node_details(
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
                inspect_details=inspect_details
            )

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

    @staticmethod
    def _get_inspects_details() -> dict:
        """Get inspects_details of containers in the current node.

        Returns:
            dict[str, dict]: container_name to inspect_details mapping.
        """
        # Get container infos in current node.
        container_names = DockerController.list_container_names()

        # Build inspect_details and return
        inspects_details = {}
        for container_name in container_names:
            inspect_details = DockerController.inspect_container(container_name)
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


class LoadImageAgent(threading.Thread):
    def __init__(
        self,
        cluster_name: str, node_name: str,
        master_hostname: str, redis_port: int,
        check_interval: int = 10
    ):
        super().__init__()
        self._cluster_name = cluster_name
        self._node_name = node_name
        self._redis_controller = RedisController(
            redis=redis.Redis(host=master_hostname, port=redis_port, encoding="utf-8", decode_responses=True)
        )

        # Other params.
        self._check_interval = check_interval
        self._is_terminated = False

    def run(self) -> None:
        """Start tracking node status and updating details.

        Returns:
            None.
        """
        while not self._is_terminated:
            start_time = time.time()
            self.load_images()
            time.sleep(max(self._check_interval - (time.time() - start_time), 0))

    def stop(self):
        self._is_terminated = True

    def load_images(self) -> None:
        """Load image from files.

        Returns:
            None.
        """

        master_details = self._redis_controller.get_master_details(cluster_name=self._cluster_name)
        node_details = self._redis_controller.get_node_details(
            cluster_name=self._cluster_name,
            node_name=self._node_name
        )
        master_image_files_details = master_details["image_files"]
        node_image_files_details = node_details["image_files"]

        # Get unloaded images
        unloaded_images = []
        for image_file, image_file_details in master_image_files_details.items():
            if image_file not in node_image_files_details:
                unloaded_images.append(image_file)
            elif (
                image_file_details["modify_time"] != node_image_files_details[image_file]["modify_time"]
                or image_file_details["size"] != node_image_files_details[image_file]["size"]
            ):
                unloaded_images.append(image_file)

        # Parallel load
        with ThreadPool(5) as pool:
            params = [
                [os.path.expanduser(f"~/.maro/clusters/{self._cluster_name}/images/{unloaded_image}")]
                for unloaded_image in unloaded_images
            ]
            pool.starmap(
                self._load_image,
                params
            )

        with NODE_DETAILS_LOCK:
            node_details = self._redis_controller.get_node_details(
                cluster_name=self._cluster_name,
                node_name=self._node_name
            )
            node_details["image_files"] = master_image_files_details
            self._redis_controller.set_node_details(
                cluster_name=self._cluster_name,
                node_name=self._node_name,
                node_details=node_details
            )

    @staticmethod
    def _load_image(image_path: str):
        logging.info(f"In loading image: {image_path}")
        DockerController.load_image(image_path=image_path)
        logging.info(f"End of loading image: {image_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(threadName)-10s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    with open(os.path.expanduser("~/.maro-local/services/maro-node-agent.config"), "r") as fr:
        node_agent_config = json.load(fr)

    node_agent = NodeAgent(
        cluster_name=node_agent_config["cluster_name"],
        node_name=node_agent_config["node_name"],
        master_hostname=node_agent_config["master_hostname"],
        redis_port=node_agent_config["redis_port"]
    )
    node_agent.start()
