# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import logging
import multiprocessing
import os
import signal
import sys
import time
from multiprocessing.pool import ThreadPool

import psutil

from ..utils.details_reader import DetailsReader
from ..utils.docker_controller import DockerController
from ..utils.params import NodeStatus
from ..utils.redis_controller import RedisController
from ..utils.resource import BasicResource
from ..utils.subprocess import Subprocess

GET_GPU_INFO_COMMAND = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits"
GET_UTILIZATION_GPUS_COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"

logger = logging.getLogger(__name__)


class NodeAgent:
    def __init__(self, local_cluster_details: dict, local_master_details: dict, local_node_details: dict):
        self._local_cluster_details = local_cluster_details
        self._local_master_details = local_master_details
        self._local_node_details = local_node_details

        self._redis_controller = RedisController(
            host=self._local_master_details["hostname"],
            port=self._local_master_details["redis"]["port"]
        )

        # Init agents.
        self.load_image_agent = LoadImageAgent(
            local_cluster_details=local_cluster_details,
            local_master_details=local_master_details,
            local_node_details=local_node_details
        )
        self.node_tracking_agent = NodeTrackingAgent(
            local_cluster_details=local_cluster_details,
            local_master_details=local_master_details,
            local_node_details=local_node_details
        )
        self.resource_tracking_agent = ResourceTrackingAgent(
            local_cluster_details=local_cluster_details,
            local_master_details=local_master_details,
            local_node_details=local_node_details
        )

        # When SIGTERM, gracefully exit.
        signal.signal(signal.SIGTERM, self.gracefully_exit)

    def start(self) -> None:
        self.init_agent()

        # Start agents.
        self.node_tracking_agent.start()
        self.load_image_agent.start()
        self.resource_tracking_agent.start()

        # Wait joins.
        self.node_tracking_agent.join()
        self.load_image_agent.join()
        self.resource_tracking_agent.join()

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
        with self._redis_controller.lock(f"lock:name_to_node_details:{self._local_node_details['name']}"):
            node_details = self._redis_controller.get_node_details(node_name=self._local_node_details["name"])
            # May be node leaving here.
            if node_details:
                node_details["state"] = state_details
                self._redis_controller.set_node_details(
                    node_name=self._local_node_details["name"],
                    node_details=node_details
                )
        sys.exit(0)

    def init_agent(self) -> None:
        self._init_resources()

    def _init_resources(self) -> None:
        node_details = self._redis_controller.get_node_details(node_name=self._local_node_details["name"])
        resources_details = node_details["resources"]

        # Get resources info
        if not isinstance(resources_details["cpu"], (float, int)):
            if resources_details["cpu"] != "all":
                logger.warning("Invalid cpu assignment, will use all cpus in this node")
            resources_details["cpu"] = psutil.cpu_count()  # (int) logical number

        if not isinstance(resources_details["memory"], (float, int)):
            if resources_details["memory"] != "all":
                logger.warning("Invalid memory assignment, will use all memories in this node")
            resources_details["memory"] = psutil.virtual_memory().total / (1024 ** 2)  # (float) in MByte

        if not isinstance(resources_details["gpu"], (float, int)):
            if resources_details["gpu"] != "all":
                logger.warning("Invalid gpu assignment, will use all gpus in this node")
            try:
                return_str = Subprocess.run(command=GET_GPU_INFO_COMMAND)
                gpus_info = return_str.split(os.linesep)
                resources_details["gpu"] = len(gpus_info) - 1  # (int) logical number
                resources_details["gpu_name"] = []
                resources_details["gpu_memory"] = []
                for info in gpus_info:
                    name, total_memory = info.split(", ")
                    resources_details["gpu_name"].append(name)
                    resources_details["gpu_memory"].append(total_memory)
            except Exception:
                resources_details["gpu"] = 0

        # Set resource details
        with self._redis_controller.lock(f"lock:name_to_node_details:{self._local_node_details['name']}"):
            node_details = self._redis_controller.get_node_details(node_name=self._local_node_details["name"])
            node_details["resources"] = resources_details
            self._redis_controller.set_node_details(
                node_name=self._local_node_details["name"],
                node_details=node_details
            )


class NodeTrackingAgent(multiprocessing.Process):
    """Node Tracking Agent.

    Update node_details by tracking container statuses and computing resource info of the current machine.
    """

    def __init__(
        self,
        local_cluster_details: dict, local_master_details: dict, local_node_details: dict,
        check_interval: int = 5
    ):
        super().__init__()
        self._local_cluster_details = local_cluster_details
        self._local_master_details = local_master_details
        self._local_node_details = local_node_details

        self._redis_controller = RedisController(
            host=self._local_master_details["hostname"],
            port=self._local_master_details["redis"]["port"]
        )

        self._check_interval = check_interval
        self._is_terminated = False

        self._container_details = {}

    def run(self) -> None:
        """Start tracking node status and updating details.

        Returns:
            None.
        """
        while not self._is_terminated:
            logger.debug("Start in NodeTrackingAgent.")
            start_time = time.time()
            self._update_details()
            time.sleep(max(self._check_interval - (time.time() - start_time), 0))
            logger.debug("End in NodeTrackingAgent.")

    def stop(self):
        self._is_terminated = True

    def _update_details(self) -> None:
        """Update details.

        Returns:
            None.
        """
        # Get or init details.

        node_details = self._redis_controller.get_node_details(node_name=self._local_node_details["name"])

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

        # State related.
        state_details = {
            "status": NodeStatus.RUNNING,
            "check_time": self._redis_controller.get_time()
        }

        # Save details.
        with self._redis_controller.lock(f"lock:name_to_node_details:{self._local_node_details['name']}"):
            node_details = self._redis_controller.get_node_details(node_name=self._local_node_details["name"])
            node_details["containers"] = name_to_container_details
            node_details["resources"] = resources_details
            node_details["state"] = state_details
            self._redis_controller.set_node_details(
                node_name=self._local_node_details["name"],
                node_details=node_details
            )

    @staticmethod
    def _update_name_to_container_details(
        container_name_to_inspect_details: dict,
        name_to_container_details: dict
    ) -> None:
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
    """Load Image Agent.

    Iterate /image_files folder of the samba server, and load image if there is a new one.
    """

    def __init__(
        self,
        local_cluster_details: dict, local_master_details: dict, local_node_details: dict,
        check_interval: int = 10
    ):
        super().__init__()
        self._local_cluster_details = local_cluster_details
        self._local_master_details = local_master_details
        self._local_node_details = local_node_details

        self._redis_controller = RedisController(
            host=self._local_master_details["hostname"],
            port=self._local_master_details["redis"]["port"]
        )

        self._check_interval = check_interval
        self._is_terminated = False

    def run(self) -> None:
        """Start tracking node status and updating details.

        Returns:
            None.
        """
        while not self._is_terminated:
            logger.debug("Start in LoadImageAgent.")
            start_time = time.time()
            self.load_images()
            time.sleep(max(self._check_interval - (time.time() - start_time), 0))
            logger.debug("End in LoadImageAgent.")

    def stop(self):
        self._is_terminated = True

    def load_images(self) -> None:
        """Load image from files.

        Returns:
            None.
        """

        master_details = self._redis_controller.get_master_details()
        node_details = self._redis_controller.get_node_details(node_name=self._local_node_details["name"])
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
                [os.path.expanduser(
                    f"~/.maro-shared/clusters/{self._local_cluster_details['name']}/image_files/{unloaded_image_name}")]
                for unloaded_image_name in unloaded_image_names
            ]
            pool.starmap(
                self._load_image,
                params
            )

        with self._redis_controller.lock(f"lock:name_to_node_details:{self._local_node_details['name']}"):
            node_details = self._redis_controller.get_node_details(node_name=self._local_node_details["name"])
            # Update with mapping in master.
            node_details["image_files"] = name_to_image_file_details_in_master
            self._redis_controller.set_node_details(
                node_name=self._local_node_details["name"],
                node_details=node_details
            )

    @staticmethod
    def _load_image(image_path: str):
        logger.info(f"In loading image: {image_path}")
        DockerController.load_image(image_path=image_path)
        logger.info(f"End of loading image: {image_path}")


class ResourceTrackingAgent(multiprocessing.Process):
    def __init__(
        self,
        local_cluster_details: dict,
        local_master_details: dict,
        local_node_details: dict,
        check_interval: int = 30
    ):
        super().__init__()
        self._local_cluster_details = local_cluster_details
        self._local_master_details = local_master_details
        self._local_node_details = local_node_details

        self._redis_controller = RedisController(
            host=self._local_master_details["hostname"],
            port=self._local_master_details["redis"]["port"]
        )

        self._check_interval = check_interval
        self._is_terminated = False

    def run(self) -> None:
        """Start tracking node status and updating details.

        Returns:
            None.
        """
        while not self._is_terminated:
            start_time = time.time()
            self.get_node_resource_usage()
            time.sleep(max(self._check_interval - (time.time() - start_time), 0))

    def get_node_resource_usage(self):
        # Get cpu usage per core.
        cpu_usage_per_core = psutil.cpu_percent(interval=self._check_interval, percpu=True)

        # Get memory usage, unit MB
        memory_usage = psutil.virtual_memory().percent / 100

        # Get nvidia-smi result.
        gpu_memory_usage = []
        try:
            return_str = Subprocess.run(command=GET_UTILIZATION_GPUS_COMMAND)
            memory_usage_per_gpu = return_str.split("\n")
            for single_usage in memory_usage_per_gpu:
                gpu_memory_usage.append(float(single_usage))
        except Exception:
            pass

        self._redis_controller.push_resource_usage(
            node_name=self._local_node_details["name"],
            cpu_usage=cpu_usage_per_core,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(threadName)-10s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    node_agent = NodeAgent(
        local_cluster_details=DetailsReader.load_local_cluster_details(),
        local_master_details=DetailsReader.load_local_master_details(),
        local_node_details=DetailsReader.load_local_node_details()
    )
    node_agent.start()
