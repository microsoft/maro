# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import logging
import multiprocessing
import os
import signal
import sys
import time
from multiprocessing.pool import ThreadPool

import psutil

from ..utils.docker_controller import DockerController
from ..utils.exception import CommandExecutionError
from ..utils.params import NodeStatus
from ..utils.redis_controller import RedisController
from ..utils.resource import BasicResource
from ..utils.subprocess import SubProcess

GET_TOTAL_GPU_COUNT_COMMAND = "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"
GET_UTILIZATION_GPUS_COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"

logger = logging.getLogger(__name__)


class NodeAgent:
    def __init__(self, cluster_name: str, node_name: str, master_hostname: str, redis_port: int):
        self._cluster_name = cluster_name
        self._node_name = node_name
        self._master_hostname = master_hostname
        self._redis_port = redis_port

        self._redis_controller = RedisController(host=master_hostname, port=redis_port)

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
        state_details = {
            "status": NodeStatus.STOPPED,
            "check_time": self._redis_controller.get_time()
        }
        with self._redis_controller.lock(f"lock:name_to_node_details:{self._node_name}"):
            node_details = self._redis_controller.get_node_details(
                cluster_name=self._cluster_name,
                node_name=self._node_name
            )
            # May be node leaving here.
            if node_details:
                node_details["state"] = state_details
                self._redis_controller.set_node_details(
                    cluster_name=self._cluster_name,
                    node_name=self._node_name,
                    node_details=node_details
                )
        sys.exit(0)

    def init_agent(self) -> None:
        self._init_resources()

    def _init_resources(self) -> None:
        node_details = self._redis_controller.get_node_details(
            cluster_name=self._cluster_name,
            node_name=self._node_name
        )
        resources_details = node_details["resources"]

        # Get resources info
        if resources_details["cpu"] == "All":
            resources_details["cpu"] = psutil.cpu_count()  # (int) logical number

        if resources_details["memory"] == "All":
            resources_details["memory"] = psutil.virtual_memory().total / 1024  # (int) in MByte

        if resources_details["gpu"] == "All":
            try:
                return_str = SubProcess.run(command=GET_TOTAL_GPU_COUNT_COMMAND)
                resources_details["gpu"] = int(return_str)  # (int) logical number
            except CommandExecutionError:
                resources_details["gpu"] = 0

        # Set resource details
        with self._redis_controller.lock(f"lock:name_to_node_details:{self._node_name}"):
            node_details = self._redis_controller.get_node_details(
                cluster_name=self._cluster_name,
                node_name=self._node_name
            )
            node_details["resources"] = resources_details
            self._redis_controller.set_node_details(
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

        self._redis_controller = RedisController(host=master_hostname, port=redis_port)

        self._check_interval = check_interval
        self._is_terminated = False

        self._container_details = {}

    def run(self) -> None:
        """Start tracking node status and updating details.

        Returns:
            None.
        """
        while not self._is_terminated:
            logger.debug(f"Start in NodeTrackingAgent.")
            start_time = time.time()
            self._update_details()
            time.sleep(max(self._check_interval - (time.time() - start_time), 0))
            logger.debug(f"End in NodeTrackingAgent.")

    def stop(self):
        self._is_terminated = True

    def _update_details(self) -> None:
        """Update details.

        Returns:
            None.
        """
        # Get or init details.

        node_details = self._redis_controller.get_node_details(
            cluster_name=self._cluster_name,
            node_name=self._node_name
        )

        # Containers related
        name_to_container_details = {}
        container_name_to_inspect_details = self._get_container_name_to_inspect_details()
        self._update_name_to_container_details(
            container_name_to_inspect_details=container_name_to_inspect_details,
            name_to_container_details=name_to_container_details
        )

        # Resources related
        resources_details = node_details["resources"]
        self._update_occupied_resources(
            container_name_to_inspect_details=container_name_to_inspect_details,
            resources_details=resources_details
        )
        self._update_actual_resources(resources_details=resources_details)

        # State related.
        state_details = {
            "status": NodeStatus.RUNNING,
            "check_time": self._redis_controller.get_time()
        }

        # Save details.
        with self._redis_controller.lock(f"lock:name_to_node_details:{self._node_name}"):
            node_details = self._redis_controller.get_node_details(
                cluster_name=self._cluster_name,
                node_name=self._node_name
            )
            node_details["containers"] = name_to_container_details
            node_details["resources"] = resources_details
            node_details["state"] = state_details
            self._redis_controller.set_node_details(
                cluster_name=self._cluster_name,
                node_name=self._node_name,
                node_details=node_details
            )

    @staticmethod
    def _update_name_to_container_details(container_name_to_inspect_details: dict,
                                          name_to_container_details: dict) -> None:
        """Update name_to_container_details from container_name_to_inspect_details.

        Args:
            container_name_to_inspect_details: Details of container inspections.
            name_to_container_details: Details of containers in the current node.

        Returns:
            None.
        """
        # Iterate container_name_to_inspect_details.
        for container_name, inspect_details in container_name_to_inspect_details.items():
            # Extract container state and labels.
            name_to_container_details[container_name] = NodeTrackingAgent._extract_labels(
                inspect_details=inspect_details
            )
            name_to_container_details[container_name]["state"] = NodeTrackingAgent._extract_state(
                inspect_details=inspect_details
            )

    @staticmethod
    def _update_occupied_resources(container_name_to_inspect_details: dict, resources_details: dict) -> None:
        """Update occupied resources from containers' inspect_details.

        Args:
            container_name_to_inspect_details: Details of container inspections.
            resources_details: Resource details of the current node.

        Returns:
            None.
        """
        # Init params.
        occupied_cpu_sum = 0.0
        occupied_memory_sum = 0.0
        occupied_gpu_sum = 0.0

        # Iterate container_name_to_inspect_details.
        for _, inspect_details in container_name_to_inspect_details.items():
            # Extract occupied resource.
            occupied_resource = NodeTrackingAgent._extract_occupied_resources(inspect_details=inspect_details)
            occupied_cpu_sum += occupied_resource.cpu
            occupied_memory_sum += occupied_resource.memory
            occupied_gpu_sum += occupied_resource.gpu

        # Update target resources.
        resources_details["target_free_cpu"] = resources_details["cpu"] - occupied_cpu_sum
        resources_details["target_free_memory"] = resources_details["memory"] - occupied_memory_sum
        resources_details["target_free_gpu"] = resources_details["gpu"] - occupied_gpu_sum

    @staticmethod
    def _update_actual_resources(resources_details: dict) -> None:
        """Update actual resources status from operating system.

        Args:
            resources_details: Resource details of the current node.

        Returns:
            None.
        """
        # Update actual cpu.
        resources_details["actual_free_cpu"] = resources_details["cpu"] - psutil.getloadavg()[0]

        # Update actual memory.
        resources_details["actual_free_memory"] = psutil.virtual_memory().free / 1024

        # Update actual gpu.
        resources_details["actual_free_gpu"] = resources_details["target_free_gpu"]
        # Get nvidia-smi result.
        try:
            return_str = SubProcess.run(command=GET_UTILIZATION_GPUS_COMMAND)
            split_str = return_str.split("\n")
            total_usage = 0
            for single_usage in split_str:
                total_usage += float(single_usage)
            resources_details["actual_gpu_usage"] = f"{float(total_usage) / len(split_str)}%"
        except CommandExecutionError:
            pass

    @staticmethod
    def _get_container_name_to_inspect_details() -> dict:
        """Get container_name_to_inspect_details of containers in the current node.

        Returns:
            dict[str, dict]: container_name to inspect_details mapping.
        """
        # Get container infos in current node.
        container_names = DockerController.list_container_names()

        # Build inspect_details and return
        container_name_to_inspect_details = {}
        for container_name in container_names:
            inspect_details = DockerController.inspect_container(container_name)
            container_name_to_inspect_details[inspect_details["Config"]["Labels"]["container_name"]] = inspect_details
        return container_name_to_inspect_details

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


class LoadImageAgent(multiprocessing.Process):
    def __init__(
        self,
        cluster_name: str, node_name: str,
        master_hostname: str, redis_port: int,
        check_interval: int = 10
    ):
        super().__init__()
        self._cluster_name = cluster_name
        self._node_name = node_name

        self._redis_controller = RedisController(host=master_hostname, port=redis_port)

        self._check_interval = check_interval
        self._is_terminated = False

    def run(self) -> None:
        """Start tracking node status and updating details.

        Returns:
            None.
        """
        while not self._is_terminated:
            logger.debug(f"Start in LoadImageAgent.")
            start_time = time.time()
            self.load_images()
            time.sleep(max(self._check_interval - (time.time() - start_time), 0))
            logger.debug(f"End in LoadImageAgent.")

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
        name_to_image_file_details_in_master = master_details["image_files"]
        name_to_image_file_details_in_node = node_details["image_files"]

        # Get unloaded images
        unloaded_image_names = []
        for image_file_name, image_file_details in name_to_image_file_details_in_master.items():
            if (
                image_file_name not in name_to_image_file_details_in_node
                or name_to_image_file_details_in_node[image_file_name]["md5_checksum"] !=
                image_file_details["md5_checksum"]
            ):
                unloaded_image_names.append(image_file_name)

        # Parallel load
        with ThreadPool(5) as pool:
            params = [
                [os.path.expanduser(f"~/.maro-shared/clusters/{self._cluster_name}/image_files/{unloaded_image_name}")]
                for unloaded_image_name in unloaded_image_names
            ]
            pool.starmap(
                self._load_image,
                params
            )

        with self._redis_controller.lock(f"lock:name_to_node_details:{self._node_name}"):
            node_details = self._redis_controller.get_node_details(
                cluster_name=self._cluster_name,
                node_name=self._node_name
            )
            # Update with mapping in master.
            node_details["image_files"] = name_to_image_file_details_in_master
            self._redis_controller.set_node_details(
                cluster_name=self._cluster_name,
                node_name=self._node_name,
                node_details=node_details
            )

    @staticmethod
    def _load_image(image_path: str):
        logger.info(f"In loading image: {image_path}")
        DockerController.load_image(image_path=image_path)
        logger.info(f"End of loading image: {image_path}")


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
