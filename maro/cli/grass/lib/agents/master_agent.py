# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import heapq
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
import uuid

from redis import Redis

from .exception import AllocationFailed, StartContainerFailed
from .resource import ContainerResource, NodeResource
from .utils import (
    delete_rejoin_container_name_to_component_name, get_containers_details, get_job_details, get_job_runtime_details,
    get_jobs_details, get_killed_job_tickets, get_node_details, get_nodes_details, get_pending_job_tickets,
    get_rejoin_component_restart_times, get_rejoin_container_name_to_component_name,
    incr_rejoin_component_restart_times, load_cluster_details, remove_killed_job_ticket, remove_pending_job_ticket,
    set_containers_details, set_job_details
)

logger = logging.getLogger(__name__)

START_CONTAINER_COMMAND = (
    "ssh -o StrictHostKeyChecking=no {admin_username}@{node_hostname} "
    "docker run "
    "-it -d "
    "--cpus {cpu} "
    "-m {memory} "
    "--name {container_name} "
    "--network host "
    "--log-driver=fluentd "
    "--log-opt tag=maro.job_id.{job_id}.container_name.{container_name} "
    "--log-opt fluentd-address={master_hostname}:{fluentd_port} "
    "-v {mount_source}:{mount_target} "
    "{environment_parameters} {labels} "
    "{image_name} {command}"
)

START_CONTAINER_WITH_GPU_COMMAND = (
    "ssh -o StrictHostKeyChecking=no {admin_username}@{node_hostname} "
    "docker run "
    "-it -d "
    "--cpus {cpu} "
    "-m {memory} "
    "--gpus {gpu} "
    "--name {container_name} "
    "--network host "
    "--log-driver=fluentd "
    "--log-opt tag=maro.job_id.{job_id}.container_name.{container_name} "
    "--log-opt fluentd-address={master_hostname}:{fluentd_port} "
    "-v {mount_source}:{mount_target} "
    "{environment_parameters} {labels} "
    "{image_name} {command}"
)

REMOVE_CONTAINER_COMMAND = (
    "ssh -o StrictHostKeyChecking=no {admin_username}@{node_hostname} "
    "docker rm -f {containers}"
)

STOP_CONTAINER_COMMAND = (
    "ssh -o StrictHostKeyChecking=no {admin_username}@{node_hostname} "
    "docker stop {containers}"
)

AVAILABLE_METRICS = {
    "cpu",
    "memory",
    "gpu"
}

ERROR_CODE_FOR_NOT_RESTART = 64
ERROR_CODE_FOR_STOP_JOB = 65
ERROR_CODES_FOR_NOT_RESTART_CONTAINER = {
    0, ERROR_CODE_FOR_NOT_RESTART, ERROR_CODE_FOR_STOP_JOB
}


class MasterAgent:
    def __init__(self, cluster_name: str):
        self._cluster_name = cluster_name
        self._cluster_details = load_cluster_details(cluster_name=cluster_name)

    def start(self) -> None:
        """Start agents.

        Returns:
            None.
        """
        job_tracking_agent = JobTrackingAgent(cluster_details=self._cluster_details)
        job_tracking_agent.start()
        container_tracking_agent = ContainerTrackingAgent(cluster_details=self._cluster_details)
        container_tracking_agent.start()
        pending_job_agent = PendingJobAgent(cluster_details=self._cluster_details)
        pending_job_agent.start()
        container_runtime_agent = ContainerRuntimeAgent(cluster_details=self._cluster_details)
        container_runtime_agent.start()
        killed_job_agent = KilledJobAgent(cluster_details=self._cluster_details)
        killed_job_agent.start()


class JobTrackingAgent(multiprocessing.Process):
    def __init__(self, cluster_details: dict, check_interval: int = 5):
        super().__init__()
        self._cluster_details = cluster_details
        self._cluster_name = cluster_details["name"]
        self._redis = Redis(
            host="localhost",
            port=cluster_details["master"]["redis"]["port"],
            charset="utf-8", decode_responses=True
        )
        self._check_interval = check_interval

    def run(self) -> None:
        """Start updating jobs_details.

        Returns:
            None.
        """
        while True:
            self._update_jobs_details()
            time.sleep(self._check_interval)

    def _update_jobs_details(self) -> None:
        """Update jobs_details with containers_details.

        Returns:
            None.
        """
        # Get details and mapping.
        containers_details = get_containers_details(
            redis=self._redis,
            cluster_name=self._cluster_name
        )
        jobs_details = get_jobs_details(
            redis=self._redis,
            cluster_name=self._cluster_name
        )
        job_id_to_job_name = self._get_job_id_to_job_name(jobs_details=jobs_details)

        # Iterate nodes details.
        for container_name, container_details in containers_details.items():
            curr_job_id = container_details["job_id"]
            if curr_job_id in job_id_to_job_name:
                curr_job_name = job_id_to_job_name[curr_job_id]
                jobs_details[curr_job_name]["containers"][container_name] = container_details
            else:
                logger.warning(f"Job Id {curr_job_id} is not found")

        # Save jobs details.
        for job_name, job_details in jobs_details.items():
            job_details["check_time"] = self._redis.time()[0]
            set_job_details(
                redis=self._redis,
                cluster_name=self._cluster_name,
                job_name=job_name,
                job_details=job_details
            )

    # Utils.

    @staticmethod
    def _get_job_id_to_job_name(jobs_details: dict) -> dict:
        """Get job_id_to_job_name mapping from jobs_details.

        Args:
            jobs_details: Details of the jobs.

        Returns:
            dict[int, str]: job_id_to_job_name mapping.
        """
        job_id_to_job_name = {}
        for job_name, job_details in jobs_details.items():
            job_id_to_job_name[job_details["id"]] = job_name
        return job_id_to_job_name


class ContainerTrackingAgent(multiprocessing.Process):
    def __init__(self, cluster_details: dict, check_interval: int = 5):
        super().__init__()
        self._cluster_details = cluster_details
        self._cluster_name = cluster_details["name"]
        self._redis = Redis(
            host="localhost",
            port=cluster_details["master"]["redis"]["port"],
            charset="utf-8", decode_responses=True
        )
        self._check_interval = check_interval

    def run(self) -> None:
        """Start updating containers_details.

        Returns:
            None.
        """
        while True:
            self._update_containers_details()
            time.sleep(self._check_interval)

    def _update_containers_details(self) -> None:
        """Update containers_details with nodes_details.

        Returns:
            None.
        """
        # Get details and init params.
        nodes_details = get_nodes_details(redis=self._redis, cluster_name=self._cluster_name)
        containers_details = {}

        # Iterate node_details.
        for _, node_details in nodes_details.items():
            containers_details.update(node_details["containers"])

        # Save containers_details.
        set_containers_details(
            redis=self._redis,
            cluster_name=self._cluster_name,
            containers_details=containers_details
        )


class ContainerRuntimeAgent(multiprocessing.Process):
    def __init__(self, cluster_details: dict, check_interval: int = 5):
        super().__init__()
        self._cluster_name = cluster_details["name"]
        self._cluster_id = cluster_details["id"]
        self._admin_username = cluster_details["user"]["admin_username"]
        self._fluentd_port = cluster_details["master"]["fluentd"]["port"]
        self._master_hostname = cluster_details["master"]["hostname"]
        self._redis = Redis(
            host="localhost",
            port=cluster_details["master"]["redis"]["port"],
            charset="utf-8", decode_responses=True
        )

        self._check_interval = check_interval

    def run(self) -> None:
        """Start tracking exited containers.

        Returns:
            None.
        """
        while True:
            self._iterate_container_status()
            time.sleep(self._check_interval)

    def _iterate_container_status(self) -> None:
        """Iterate container status.

        Find the exited container and try to restart it if the rule exists.

        Returns:
            None.
        """
        # Get details.
        containers_details = get_containers_details(
            redis=self._redis,
            cluster_name=self._cluster_name
        )

        # Iterate container status.
        for container_name, container_details in containers_details.items():
            # Get job_runtime_details and flags.
            job_runtime_details = get_job_runtime_details(
                redis=self._redis,
                job_id=container_details["job_id"]
            )

            # Remove container.
            is_remove_container = self._is_remove_container(
                container_details=container_details,
                job_runtime_details=job_runtime_details
            )
            if is_remove_container:
                self._remove_container(container_name=container_name, container_details=container_details)

            # Restart container.
            if self._is_restart_container(
                container_details=container_details,
                job_runtime_details=job_runtime_details
            ):
                self._restart_container(container_name=container_name, container_details=container_details)

            # Stop job.
            if self._is_stop_job(container_details=container_details):
                self._stop_job(job_id=container_details["job_id"], is_remove_container=is_remove_container)

    @staticmethod
    def _is_remove_container(container_details: dict, job_runtime_details: dict) -> bool:
        """Check if the container need to be removed.

        Args:
            container_details (dict): Details of the container.
            job_runtime_details (dict): Runtime details of the job.

        Returns:
            bool: True or False.
        """
        return (
            container_details["state"]["Status"] == "exited"
            and job_runtime_details is not None
            and job_runtime_details.get("is_remove_failed_container") == "1"
        )

    def _is_restart_container(self, container_details: dict, job_runtime_details: dict) -> bool:
        """Check if the container need to be removed.

        Args:
            container_details (dict): Details of the container.
            job_runtime_details (dict): Runtime details of the job.

        Returns:
            bool: True or False.
        """
        exceed_maximum_restart_times = get_rejoin_component_restart_times(
            self._redis,
            job_id=container_details["job_id"],
            component_id=container_details["component_id"]
        ) >= int(job_runtime_details.get("rejoin:max_restart_times", sys.maxsize))
        return (
            container_details["state"]["Status"] == "exited"
            and container_details["state"]["ExitCode"] not in ERROR_CODES_FOR_NOT_RESTART_CONTAINER
            and job_runtime_details is not None
            and job_runtime_details.get("rejoin:enable") == "1"
            and not exceed_maximum_restart_times
        )

    @staticmethod
    def _is_stop_job(container_details: dict) -> bool:
        """Check if the job need to be stop.

        Args:
            container_details (dict): Details of the container.

        Returns:
            bool: True of False.
        """
        return (
            container_details["state"]["Status"] == "exited"
            and container_details["state"]["ExitCode"] == ERROR_CODE_FOR_STOP_JOB
        )

    def _restart_container(self, container_name: str, container_details: dict) -> None:
        """Restart container.

        Args:
            container_name (str): Name of the exited container.
            container_details (dict): Details of the exited container.

        Returns:
            None.
        """
        # Get component_name_to_container_name.
        rejoin_container_name_to_component_name = get_rejoin_container_name_to_component_name(
            redis=self._redis,
            job_id=container_details["job_id"]
        )

        # If the mapping not exists, or the container is not in the mapping, skip the restart operation.
        if (
            rejoin_container_name_to_component_name is None or
            container_name not in rejoin_container_name_to_component_name
        ):
            logger.warning(f"Container {container_name} is not found in container_name_to_component_name mapping")
            return
        else:
            try:
                # Get params.
                component_name = rejoin_container_name_to_component_name[container_name]

                # Get resources and allocation plan.
                free_resources = ResourceManagementExecutor.get_free_resources(
                    redis=self._redis,
                    cluster_name=self._cluster_name
                )
                required_resources = [
                    ContainerResource(
                        container_name=ResourceManagementExecutor.build_container_name(
                            job_id=container_details["job_id"],
                            component_id=container_details["component_id"],
                            component_index=container_details["component_index"]
                        ),
                        cpu=float(container_details["cpu"]),
                        memory=float(container_details["memory"].replace("m", "")),
                        gpu=float(container_details["gpu"])
                    )
                ]
                allocation_plan = ResourceManagementExecutor.get_single_metric_balanced_allocation_plan(
                    allocation_details={"metric": "cpu"},
                    required_resources=required_resources,
                    free_resources=free_resources
                )

                # Start a new container.
                job_details = get_job_details(
                    redis=self._redis,
                    cluster_name=self._cluster_name,
                    job_name=container_details["job_name"]
                )
                for container_name, node_name in allocation_plan.items():
                    node_details = get_node_details(
                        redis=self._redis,
                        cluster_name=self._cluster_name,
                        node_name=node_name
                    )
                    self._start_container(
                        container_name=container_name,
                        node_details=node_details,
                        job_details=job_details,
                        component_name=component_name
                    )
                incr_rejoin_component_restart_times(
                    redis=self._redis,
                    job_id=container_details["job_id"],
                    component_id=container_details["component_id"]
                )
            except AllocationFailed as e:
                logger.warning(f"Allocation failed with {e}")
            except StartContainerFailed as e:
                logger.warning(f"Start container failed with {e}")

    def _remove_container(self, container_name: str, container_details: dict) -> None:
        """Remove container.

        Args:
            container_name (str): Name of the container.
            container_details (dict): Details of the container.

        Returns:
            None.
        """
        # Get details and params.
        node_name = container_details["node_name"]
        node_details = get_node_details(
            redis=self._redis,
            cluster_name=self._cluster_name,
            node_name=node_name
        )

        # Load and exec command.
        command = REMOVE_CONTAINER_COMMAND.format(
            admin_username=self._admin_username,
            node_hostname=node_details["hostname"],
            containers=container_name
        )
        completed_process = subprocess.run(
            command,
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
        )
        if completed_process.returncode != 0:
            logger.error(f"No container {container_name} in {node_name}")

    def _stop_job(self, job_id: str, is_remove_container: bool) -> None:
        """Stop job.

        Args:
            job_id (str): Id of the job.
            is_remove_container (bool): If the containers need to be removed.

        Returns:
            None.
        """
        # Delete mapping if fault tolerance is activated.
        delete_rejoin_container_name_to_component_name(
            redis=self._redis,
            job_id=job_id
        )

        # Load details and vars.
        nodes_details = get_nodes_details(
            redis=self._redis,
            cluster_name=self._cluster_name
        )

        # Delete containers.
        for node_name, node_details in nodes_details.items():
            # Load details.
            container_details = node_details["containers"]
            node_hostname = node_details["hostname"]

            # Filter containers.
            stoppable_containers = []
            for container_name in container_details:
                if container_name.startswith(job_id):
                    stoppable_containers.append(container_name)

            # Stop containers.
            if len(stoppable_containers) > 0:
                if is_remove_container:
                    command = REMOVE_CONTAINER_COMMAND.format(
                        admin_username=self._admin_username,
                        node_hostname=node_hostname,
                        containers=" ".join(stoppable_containers)
                    )
                else:
                    command = STOP_CONTAINER_COMMAND.format(
                        admin_username=self._admin_username,
                        node_hostname=node_hostname,
                        containers=" ".join(stoppable_containers)
                    )
                completed_process = subprocess.run(
                    command,
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    encoding="utf8"
                )
                if completed_process.returncode != 0:
                    logger.error(completed_process.stderr)
                logger.info(command)

    def _start_container(self, container_name: str, node_details: dict, job_details: dict, component_name: str) -> None:
        """Start container.

        Args:
            container_name: Name of the container.
            node_details: Details of the node.
            job_details: Details of the job.
            component_name: Name of the component from mapping.

        Returns:
            None.
        """
        # Get mapping.
        component_id_to_component_type = JobExecutor.get_component_id_to_component_type(job_details=job_details)

        # Parse params.
        cluster_name = self._cluster_name
        cluster_id = self._cluster_id
        node_id = node_details["id"]
        node_name = node_details["name"]
        job_id = job_details["id"]
        job_name = job_details["name"]
        component_id = container_name.split("-")[1]
        component_index = container_name.split("-")[2]
        component_type = component_id_to_component_type[component_id]
        cpu = job_details["components"][component_type]["resources"]["cpu"]
        memory = job_details["components"][component_type]["resources"]["memory"]
        gpu = job_details["components"][component_type]["resources"]["gpu"]

        # Parse environment parameters and labels.
        environment_parameters = (
            f"-e CLUSTER_ID={cluster_id} "
            f"-e CLUSTER_NAME={cluster_name} "
            f"-e NODE_ID={node_id} "
            f"-e NODE_NAME={node_name} "
            f"-e JOB_ID={job_id} "
            f"-e JOB_NAME={job_name} "
            f"-e COMPONENT_ID={component_id} "
            f"-e COMPONENT_TYPE={component_type} "
            f"-e COMPONENT_INDEX={component_index} "
            f"-e CONTAINER_NAME={container_name} "
            f"-e PYTHONUNBUFFERED=0 "
            f"-e COMPONENT_NAME={component_name}"
        )
        labels = (
            f"-l cluster_id={cluster_id} "
            f"-l cluster_name={cluster_name} "
            f"-l node_id={node_id} "
            f"-l node_name={node_name} "
            f"-l job_id={job_id} "
            f"-l job_name={job_name} "
            f"-l component_type={component_type} "
            f"-l component_id={component_id} "
            f"-l component_index={component_index} "
            f"-l container_name={container_name} "
            f"-l cpu={cpu} "
            f"-l memory={memory} "
            f"-l gpu={gpu}"
        )

        # Load command.
        if job_details["components"][component_type]["resources"]["gpu"] != 0:
            command = START_CONTAINER_WITH_GPU_COMMAND
        else:
            command = START_CONTAINER_COMMAND
        command = command.format(
            # cluster related.
            admin_username=self._admin_username,
            master_hostname=self._master_hostname,
            node_hostname=node_details["hostname"],
            fluentd_port=self._fluentd_port,

            # job related (user).
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            mount_target=job_details["components"][component_type]["mount"]["target"],
            command=job_details["components"][component_type]["command"],
            image_name=job_details["components"][component_type]["image"],

            # job related (system).
            container_name=container_name,
            job_id=job_id,
            mount_source=f"~/.maro/clusters/{cluster_name}/data/",
            environment_parameters=environment_parameters,
            labels=labels
        )

        # Exec command.
        logger.info(command)
        completed_process = subprocess.run(
            command,
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
        )
        if completed_process.returncode != 0:
            raise AllocationFailed(completed_process.stderr)


class PendingJobAgent(multiprocessing.Process):
    def __init__(self, cluster_details: dict, check_interval: int = 5):
        super().__init__()
        self._cluster_name = cluster_details["name"]
        self._cluster_id = cluster_details["id"]
        self._admin_username = cluster_details["user"]["admin_username"]
        self._fluentd_port = cluster_details["master"]["fluentd"]["port"]
        self._master_hostname = cluster_details["master"]["hostname"]
        self._redis = Redis(
            host="localhost",
            port=cluster_details["master"]["redis"]["port"],
            charset="utf-8", decode_responses=True
        )

        self._check_interval = check_interval

        self._pending_jobs = []

    def run(self) -> None:
        """Start tracking pending job tickets.

        Returns:
            None.
        """
        while True:
            self._schedule_pending_job_tickets()
            time.sleep(self._check_interval)

    def _schedule_pending_job_tickets(self) -> None:
        """Schedule pending job tickets.

        Returns:
            None.
        """
        # Get tickets.
        self._pending_jobs = get_pending_job_tickets(
            redis=self._redis,
            cluster_name=self._cluster_name
        )

        # Iterate tickets.
        for pending_job_name in self._pending_jobs:
            # Get details.
            job_details = get_job_details(
                redis=self._redis,
                cluster_name=self._cluster_name,
                job_name=pending_job_name
            )

            # Get resources info.
            free_resources = ResourceManagementExecutor.get_free_resources(
                redis=self._redis,
                cluster_name=self._cluster_name
            )
            required_resources = ResourceManagementExecutor.get_required_resources(job_details=job_details)

            # Do allocation and start job.
            try:
                allocation_plan = ResourceManagementExecutor.get_allocation_plan(
                    allocation_details=job_details["allocation"],
                    required_resources=required_resources,
                    free_resources=free_resources
                )
                for container_name, node_name in allocation_plan.items():
                    node_details = get_node_details(
                        redis=self._redis,
                        cluster_name=self._cluster_name,
                        node_name=node_name
                    )
                    self._start_container(
                        container_name=container_name,
                        node_details=node_details,
                        job_details=job_details
                    )
                remove_pending_job_ticket(
                    redis=self._redis,
                    cluster_name=self._cluster_name,
                    job_name=pending_job_name
                )
            except AllocationFailed as e:
                logger.warning(f"Allocation failed with {e}")
            except StartContainerFailed as e:
                remove_pending_job_ticket(
                    redis=self._redis,
                    cluster_name=self._cluster_name,
                    job_name=pending_job_name
                )
                logger.warning(f"Start container failed with {e}")

    def _start_container(self, container_name: str, node_details: dict, job_details: dict):
        """Start container.

        Args:
            container_name: Name of the container.
            node_details: Details of the node.
            job_details: Details of the job.

        Returns:
            None.
        """
        # Get mapping.
        component_id_to_component_type = JobExecutor.get_component_id_to_component_type(job_details=job_details)

        # Parse params.
        cluster_name = self._cluster_name
        cluster_id = self._cluster_id
        node_id = node_details["id"]
        node_name = node_details["name"]
        job_name = job_details["name"]
        job_id = job_details["id"]
        component_id = container_name.split("-")[1]
        component_index = container_name.split("-")[2]
        component_type = component_id_to_component_type[component_id]
        cpu = job_details["components"][component_type]["resources"]["cpu"]
        memory = job_details["components"][component_type]["resources"]["memory"]
        gpu = job_details["components"][component_type]["resources"]["gpu"]

        # Parse environment parameters and labels.
        environment_parameters = (
            f"-e CLUSTER_ID={cluster_id} "
            f"-e CLUSTER_NAME={cluster_name} "
            f"-e NODE_ID={node_id} "
            f"-e NODE_NAME={node_name} "
            f"-e JOB_ID={job_id} "
            f"-e JOB_NAME={job_name} "
            f"-e COMPONENT_ID={component_id} "
            f"-e COMPONENT_TYPE={component_type} "
            f"-e COMPONENT_INDEX={component_index} "
            f"-e CONTAINER_NAME={container_name} "
            f"-e PYTHONUNBUFFERED=0"
        )
        labels = (
            f"-l cluster_id={cluster_id} "
            f"-l cluster_name={cluster_name} "
            f"-l node_id={node_id} "
            f"-l node_name={node_name} "
            f"-l job_id={job_id} "
            f"-l job_name={job_name} "
            f"-l component_type={component_type} "
            f"-l component_id={component_id} "
            f"-l component_index={component_index} "
            f"-l container_name={container_name} "
            f"-l cpu={cpu} "
            f"-l memory={memory} "
            f"-l gpu={gpu}"
        )

        # Load command.
        if job_details["components"][component_type]["resources"]["gpu"] != 0:
            command = START_CONTAINER_WITH_GPU_COMMAND
        else:
            command = START_CONTAINER_COMMAND
        command = command.format(
            # cluster related.
            admin_username=self._admin_username,
            master_hostname=self._master_hostname,
            node_hostname=node_details["hostname"],
            fluentd_port=self._fluentd_port,

            # job related (user).
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            mount_target=job_details["components"][component_type]["mount"]["target"],
            command=job_details["components"][component_type]["command"],
            image_name=job_details["components"][component_type]["image"],

            # job related (system).
            container_name=container_name,
            job_id=job_id,
            mount_source=f"~/.maro/clusters/{cluster_name}/data/",
            environment_parameters=environment_parameters,
            labels=labels
        )

        # Exec command.
        logger.info(command)
        completed_process = subprocess.run(
            command,
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
        )
        if completed_process.returncode != 0:
            raise AllocationFailed(completed_process.stderr)


class KilledJobAgent(multiprocessing.Process):
    def __init__(self, cluster_details: dict, check_interval: int = 5):
        super().__init__()
        self._cluster_name = cluster_details["name"]
        self._cluster_id = cluster_details["id"]
        self._admin_username = cluster_details["user"]["admin_username"]
        self._redis = Redis(
            host="localhost",
            port=cluster_details["master"]["redis"]["port"],
            charset="utf-8", decode_responses=True
        )

        self._check_interval = check_interval

        self._killed_job_tickets = []

    def run(self) -> None:
        """Start tracking killed job tickets.

        Returns:
            None.
        """
        while True:
            self._schedule_killed_job_tickets()
            time.sleep(self._check_interval)

    def _schedule_killed_job_tickets(self):
        """Schedule killed job tickets.

        Returns:
            None.
        """
        # Get tickets.
        self._killed_job_tickets = get_killed_job_tickets(
            redis=self._redis,
            cluster_name=self._cluster_name
        )

        # Iterate tickets.
        for job_name in self._killed_job_tickets:
            # Get details.
            job_details = get_job_details(
                redis=self._redis,
                cluster_name=self._cluster_name,
                job_name=job_name
            )
            if job_details is not None:
                # Kill job.
                self._kill_job(job_details=job_details)
            else:
                logger.warning(f"{job_name} not exists, cannot be stopped")

            # Remove killed job ticket.
            remove_killed_job_ticket(
                redis=self._redis,
                cluster_name=self._cluster_name,
                job_name=job_name
            )

    def _kill_job(self, job_details: dict) -> None:
        """Kill job and stop containers.

        Args:
            job_details (dict): Details of the job.

        Returns:
            None.
        """
        # Get params.
        job_id = job_details["id"]

        # Delete mapping if fault tolerance is activated.
        delete_rejoin_container_name_to_component_name(
            redis=self._redis,
            job_id=job_id
        )

        # Load details and vars.
        nodes_details = get_nodes_details(
            redis=self._redis,
            cluster_name=self._cluster_name
        )

        # Delete containers.
        for node_name, node_details in nodes_details.items():
            # Load details.
            container_details = node_details["containers"]
            node_hostname = node_details["hostname"]

            # Filter containers.
            removable_containers = []
            for container_name in container_details:
                if container_name.startswith(job_id):
                    removable_containers.append(container_name)

            # Stop containers.
            if len(removable_containers) > 0:
                command = STOP_CONTAINER_COMMAND.format(
                    admin_username=self._admin_username,
                    node_hostname=node_hostname,
                    containers=" ".join(removable_containers)
                )
                completed_process = subprocess.run(
                    command,
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    encoding="utf8"
                )
                if completed_process.returncode != 0:
                    logger.error(completed_process.stderr)
                logger.info(command)


class ResourceManagementExecutor:
    @staticmethod
    def get_allocation_plan(allocation_details: dict, required_resources: list, free_resources: list) -> dict:
        """Get container allocation mapping.

        Args:
            allocation_details (dict): Details of allocation config.
            required_resources (list): List of ContainerResource.
            free_resources (list): List of NodeResource.

        Returns:
            dict: container_name to node_name mapping.
        """
        if allocation_details["mode"] == "single-metric-balanced":
            return ResourceManagementExecutor.get_single_metric_balanced_allocation_plan(
                allocation_details=allocation_details,
                required_resources=required_resources,
                free_resources=free_resources
            )
        elif allocation_details["mode"] == "single-metric-compacted":
            return ResourceManagementExecutor.get_single_metric_compacted_allocation_plan(
                allocation_details=allocation_details,
                required_resources=required_resources,
                free_resources=free_resources
            )
        else:
            raise AllocationFailed("Invalid allocation mode")

    @staticmethod
    def get_single_metric_compacted_allocation_plan(
        allocation_details: dict,
        required_resources: list, free_resources: list
    ) -> dict:
        """Get single_metric_compacted allocation plan.

        The strategy uses a specific metric as the priority,
        then use a greedy approach to match the container to the available node
        with the smallest remaining free resource.

        Args:
            allocation_details (dict): Details of allocation config.
            required_resources (list): List of ContainerResource.
            free_resources (list): List of NodeResource.

        Returns:
            dict[str, str]: container_name to node_name mapping.
        """
        # Init params.
        allocation_plan = {}
        if "metric" not in allocation_details or allocation_details["metric"].lower() not in AVAILABLE_METRICS:
            raise AllocationFailed("Invalid allocation parameter: metric")
        metric = allocation_details["metric"].lower()

        # Init resources PQ.
        required_resources_pq = []
        for required_resource in required_resources:
            heapq.heappush(
                required_resources_pq,
                (-getattr(required_resource, metric), required_resource)
            )
        free_resources_pq = []
        for free_resource in free_resources:
            heapq.heappush(
                free_resources_pq,
                (getattr(free_resource, metric), free_resource)
            )

        # Get allocation.
        while len(required_resources_pq) > 0:
            is_allocated = False

            # Get vars.
            required_resource = heapq.heappop(required_resources_pq)[1]
            free_resource = None

            not_usable_free_resources = []
            while len(free_resources_pq) > 0:
                free_resource = heapq.heappop(free_resources_pq)[1]
                if free_resource >= required_resource:
                    is_allocated = True
                    break
                else:
                    not_usable_free_resources.append(free_resource)

            # Do allocation or return error.
            if is_allocated:
                allocation_plan[required_resource.container_name] = free_resource.node_name
                free_resource.cpu -= required_resource.cpu
                free_resource.memory -= required_resource.memory
                free_resource.gpu -= required_resource.gpu
                heapq.heappush(
                    free_resources_pq,
                    (getattr(free_resource, metric), free_resource)
                )
                for not_usable_free_resource in not_usable_free_resources:
                    heapq.heappush(
                        free_resources_pq,
                        (getattr(not_usable_free_resource, metric), not_usable_free_resource)
                    )
            else:
                # add previous resources back, to do printing.
                for not_usable_free_resource in not_usable_free_resources:
                    heapq.heappush(
                        free_resources_pq,
                        (getattr(not_usable_free_resource, metric), not_usable_free_resource)
                    )
                heapq.heappush(
                    required_resources_pq,
                    (-getattr(required_resource, metric), required_resource)
                )

                logger.warning(allocation_plan)
                logger.warning(required_resources_pq)
                logger.warning(free_resources_pq)
                raise AllocationFailed("Unable to allocate, Abort")

        logger.info(required_resources)
        logger.info(free_resources)
        return allocation_plan

    @staticmethod
    def get_single_metric_balanced_allocation_plan(
        allocation_details: dict,
        required_resources: list, free_resources: list
    ) -> dict:
        """Get single_metric_balanced allocation plan.

        The strategy uses a specific metric as the priority,
        then use a greedy approach to match the container to the available node
        with the largest remaining free resource.

        Args:
            allocation_details (dict): Details of allocation config.
            required_resources (list): List of ContainerResource.
            free_resources (list): List of NodeResource.

        Returns:
            dict[str, str]: container_name to node_name mapping.
        """
        # Init params.
        allocation_plan = {}
        if "metric" not in allocation_details or allocation_details["metric"].lower() not in AVAILABLE_METRICS:
            raise AllocationFailed("Invalid allocation parameter: metric")
        metric = allocation_details["metric"].lower()

        # Init resources PQ.
        required_resources_pq = []
        for required_resource in required_resources:
            heapq.heappush(
                required_resources_pq,
                (-getattr(required_resource, metric), required_resource)
            )
        free_resources_pq = []
        for free_resource in free_resources:
            heapq.heappush(
                free_resources_pq,
                (-getattr(free_resource, metric), free_resource)
            )

        # Get allocation.
        while len(required_resources_pq) > 0:
            # Get list, not tuple.
            required_resource = heapq.heappop(required_resources_pq)[1]

            not_usable_free_resources = []
            is_allocated = False
            free_resource = None
            while len(free_resources_pq) > 0:
                # Get list, not tuple.
                free_resource = heapq.heappop(free_resources_pq)[1]
                if free_resource >= required_resource:
                    is_allocated = True
                    break
                else:
                    not_usable_free_resources.append(free_resource)

            # Do allocation or return error.
            if is_allocated:
                allocation_plan[required_resource.container_name] = free_resource.node_name
                free_resource.cpu -= required_resource.cpu
                free_resource.memory -= required_resource.memory
                free_resource.gpu -= required_resource.gpu
                heapq.heappush(
                    free_resources_pq,
                    (-getattr(free_resource, metric), free_resource)
                )
                for not_usable_free_resource in not_usable_free_resources:
                    heapq.heappush(
                        free_resources_pq,
                        (-getattr(not_usable_free_resource, metric), not_usable_free_resource)
                    )
            else:
                # add previous resources back, to do printing.
                for not_usable_free_resource in not_usable_free_resources:
                    heapq.heappush(
                        free_resources_pq,
                        (-getattr(not_usable_free_resource, metric), not_usable_free_resource)
                    )
                heapq.heappush(
                    required_resources_pq,
                    (-getattr(required_resource, metric), required_resource)
                )

                logger.warning(allocation_plan)
                logger.warning(required_resources_pq)
                logger.warning(free_resources_pq)
                raise AllocationFailed("Unable to allocate, Abort")

        logger.info(required_resources)
        logger.info(free_resources)
        return allocation_plan

    @staticmethod
    def get_free_resources(redis: Redis, cluster_name: str) -> list:
        """Get free resources of nodes in cluster.

        Args:
            redis (Redis): Redis Client of current cluster.
            cluster_name (str): Name of the cluster.

        Returns:
            list: List of NodeResource.
        """
        # Load details.
        nodes_details = get_nodes_details(
            redis=redis,
            cluster_name=cluster_name
        )

        # Get free resources.
        free_resources_list = []
        for node_name, node_details in nodes_details.items():
            target_free_cpu = node_details["resources"]["target_free_cpu"]
            target_free_memory = node_details["resources"]["target_free_memory"]
            target_free_gpu = node_details["resources"]["target_free_gpu"]

            if node_details["state"] == "Running":
                free_resources_list.append(
                    NodeResource(
                        node_name=node_name,
                        cpu=target_free_cpu,
                        memory=target_free_memory,
                        gpu=target_free_gpu
                    )
                )
        return free_resources_list

    @staticmethod
    def get_required_resources(job_details: dict) -> list:
        """Get required resources from job_details.

        Args:
            job_details: Details of jobs.

        Returns:
            list: List of ContainerResource.
        """
        # Load configs.
        components_details = job_details["components"]
        job_id = job_details["id"]

        # Get required resources.
        resources_list = []
        for component_type, component_details in components_details.items():
            component_id = component_details["id"]
            component_num = component_details["num"]
            required_cpu = component_details["resources"]["cpu"]
            required_memory = int(component_details["resources"]["memory"].replace("m", ""))
            required_gpu = component_details["resources"]["gpu"]

            for i in range(component_num):
                resources_list.append(
                    ContainerResource(
                        container_name=ResourceManagementExecutor.build_container_name(job_id, component_id, i),
                        cpu=required_cpu,
                        memory=required_memory,
                        gpu=required_gpu,
                    )
                )
        return resources_list

    @staticmethod
    def build_container_name(job_id: str, component_id: str, component_index: int) -> str:
        """Build the container name with job-related params.

        Ref: The container name must be from 1 to 255 characters long.

        Args:
            job_id: The Id of the job.
            component_id: The Id of the component.
            component_index: The index of the current component.

        Returns:
            str: Name of the container.
        """
        return f"{job_id}-{component_id}-{component_index}-{uuid.uuid4().hex[:6]}"


class JobExecutor:
    @staticmethod
    def get_component_id_to_component_type(job_details: dict) -> dict:
        """Get component_id_to_component_type mapping from job_details

        Args:
            job_details: Details of jobs.

        Returns:
            dict[str, str]: component_id_to_component_type mapping.
        """
        # Load details.
        components_details = job_details["components"]

        # Get component_id_to_type.
        component_id_to_component_type = {}
        for component_type, component_details in components_details.items():
            component_id_to_component_type[component_details["id"]] = component_type
        return component_id_to_component_type


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(levelname)-7s] - %(asctime)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    with open(os.path.expanduser("~/.maro-local/agents/master_agent.config"), "r") as fr:
        master_agent_config = json.load(fr)

    master_agent = MasterAgent(
        cluster_name=master_agent_config["cluster_name"]
    )
    master_agent.start()
