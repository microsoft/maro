# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import heapq
import json
import logging
import multiprocessing
import os
import subprocess
import time

import redis

from .utils import (
    get_job_details, get_jobs_details, get_killed_jobs, get_node_details, get_nodes_details, get_pending_jobs,
    load_cluster_details, remove_killed_job, remove_pending_job, set_job_details
)

logger = logging.getLogger(__name__)

START_CONTAINER_COMMAND = """\
ssh -o StrictHostKeyChecking=no {admin_username}@{node_hostname} \
docker run \
-it -d \
--cpus {cpu} \
-m {memory} \
--name {container_name} \
--network host \
--log-driver=fluentd \
--log-opt tag=maro.job_id.{job_id}.container_name.{container_name} \
--log-opt fluentd-address={master_hostname}:{fluentd_port} \
--label required_cpu={cpu} \
--label required_memory={memory} \
--label required_gpu={gpu} \
-v {mount_source}:{mount_target} \
{environment_parameters} \
{image_name} {command}\
"""

START_CONTAINER_WITH_GPU_COMMAND = """\
ssh -o StrictHostKeyChecking=no {admin_username}@{node_hostname} \
docker run \
-it -d \
--cpus {cpu} \
-m {memory} \
--gpus {gpu} \
--name {container_name} \
--network host \
--log-driver=fluentd \
--log-opt tag=maro.job_id.{job_id}.container_name.{container_name} \
--log-opt fluentd-address={master_hostname}:{fluentd_port} \
--label required_cpu={cpu} \
--label required_memory={memory} \
--label required_gpu={gpu} \
-v {mount_source}:{mount_target} \
{environment_parameters} \
{image_name} {command}\
"""

REMOVE_CONTAINER_COMMAND = """\
ssh -o StrictHostKeyChecking=no {admin_username}@{node_hostname} \
docker rm {containers}\
"""

STOP_CONTAINER_COMMAND = """\
ssh -o StrictHostKeyChecking=no {admin_username}@{node_hostname} \
docker stop {containers}\
"""

METRIC_TO_INDEX = {
    'cpu': 1,
    'memory': 2,
    'gpu': 3
}


class MasterAgent:
    def __init__(self, cluster_name: str, redis_port: int):
        self._cluster_name = cluster_name
        self._redis_port = redis_port

    def start(self) -> None:
        job_tracking_agent = JobTrackingAgent(
            cluster_name=self._cluster_name,
            redis_port=self._redis_port)
        job_tracking_agent.start()
        pending_job_agent = PendingJobAgent(
            cluster_name=self._cluster_name,
            redis_port=self._redis_port)
        pending_job_agent.start()
        killed_job_agent = KilledJobAgent(
            cluster_name=self._cluster_name,
            redis_port=self._redis_port)
        killed_job_agent.start()


class JobTrackingAgent(multiprocessing.Process):
    def __init__(self, cluster_name: str, redis_port: int, check_interval: int = 10):
        super().__init__()
        self._cluster_name = cluster_name
        self._redis = redis.Redis(
            host='localhost',
            port=redis_port,
            charset="utf-8", decode_responses=True
        )

        self._check_interval = check_interval

    def run(self) -> None:
        while True:
            self._update_jobs_details()
            time.sleep(self._check_interval)

    def _update_jobs_details(self) -> None:
        # Get details
        nodes_details = get_nodes_details(
            redis=self._redis,
            cluster_name=self._cluster_name
        )
        jobs_details = get_jobs_details(
            redis=self._redis,
            cluster_name=self._cluster_name
        )
        job_id_to_job_name = self._get_job_id_to_job_name()

        # Iterate nodes details
        for node_name, node_details in nodes_details.items():
            containers_details = node_details['containers']

            for container_name, container_details in containers_details.items():
                curr_job_id = container_name.split('-')[0]
                if curr_job_id in job_id_to_job_name:
                    curr_job_name = job_id_to_job_name[curr_job_id]
                    jobs_details[curr_job_name]['containers'][container_name] = container_details
                else:
                    logger.warning(f"Job Id {curr_job_id} is not found")

        # Save jobs details
        for job_name, job_details in jobs_details.items():
            job_details['check_time'] = self._redis.time()[0]
            set_job_details(
                redis=self._redis,
                cluster_name=self._cluster_name,
                job_name=job_name,
                job_details=job_details
            )

    # Utils

    def _get_job_id_to_job_name(self):
        # Get details
        jobs_details = get_jobs_details(
            redis=self._redis,
            cluster_name=self._cluster_name
        )

        # Iterate job_details
        job_id_to_job_name = {}
        for job_name, job_details in jobs_details.items():
            job_id_to_job_name[job_details['id']] = job_name

        return job_id_to_job_name


class PendingJobAgent(multiprocessing.Process):
    def __init__(self, cluster_name: str, redis_port: int, check_interval: int = 10):
        super().__init__()
        self._cluster_name = cluster_name
        self._redis = redis.Redis(
            host='localhost',
            port=redis_port,
            charset="utf-8", decode_responses=True
        )
        cluster_details = load_cluster_details(cluster_name=cluster_name)
        self._cluster_id = cluster_details['id']
        self._admin_username = cluster_details['user']['admin_username']
        self._fluentd_port = cluster_details['master']['fluentd']['port']
        self._master_hostname = cluster_details['master']['hostname']

        self._check_interval = check_interval

        self._pending_jobs = []

    def run(self) -> None:
        while True:
            self._schedule_pending_jobs()
            time.sleep(self._check_interval)

    def _schedule_pending_jobs(self):
        self._pending_jobs = get_pending_jobs(
            redis=self._redis,
            cluster_name=self._cluster_name
        )

        for pending_job in self._pending_jobs:
            # Get details
            job_details = get_job_details(
                redis=self._redis,
                cluster_name=self._cluster_name,
                job_name=pending_job
            )

            # Get resources info
            free_resources = self._get_free_resources()
            required_resources = self._get_required_resources(
                job_details=job_details)

            # Do allocation and start job
            try:
                allocation_plan = self._get_allocation_plan(
                    allocation_details=job_details['allocation'],
                    required_resources=required_resources,
                    free_resources=free_resources
                )
                for container_name, node_name in allocation_plan.items():
                    self.remove_container(
                        container_name=container_name)
                    self.start_container(
                        container_name=container_name,
                        node_name=node_name,
                        job_details=job_details)
                remove_pending_job(
                    redis=self._redis,
                    cluster_name=self._cluster_name,
                    job_name=pending_job
                )
            except AllocationFailed as e:
                logger.warning(
                    f"Allocation failed with {e}"
                )
            except StartContainerFailed as e:
                remove_pending_job(
                    redis=self._redis,
                    cluster_name=self._cluster_name,
                    job_name=pending_job
                )
                logger.warning(
                    f"Start container failed with {e}"
                )

    def _get_allocation_plan(self, allocation_details: dict, required_resources: list, free_resources: list) -> dict:
        allocation = allocation_details

        if allocation['mode'] == 'single-metric-balanced':
            return self._get_single_metric_balanced_allocation_plan(
                allocation_details=allocation_details,
                required_resources=required_resources,
                free_resources=free_resources
            )
        elif allocation['mode'] == 'single-metric-compacted':
            return self._get_single_metric_compacted_allocation_plan(
                allocation_details=allocation_details,
                required_resources=required_resources,
                free_resources=free_resources
            )
        else:
            raise AllocationFailed("Invalid allocation mode")

    @staticmethod
    def _get_single_metric_compacted_allocation_plan(
            allocation_details: dict, required_resources: list, free_resources: list) -> dict:
        # Init params
        allocation_plan = {}
        if 'metric' not in allocation_details or allocation_details['metric'].lower() not in METRIC_TO_INDEX:
            raise AllocationFailed("Invalid allocation parameter: metric")
        metric_index = METRIC_TO_INDEX[allocation_details['metric'].lower()]

        # Init reverse PQ
        required_resources_pq = []
        for required_resource in required_resources:
            heapq.heappush(
                required_resources_pq,
                (-required_resource[metric_index], required_resource)
            )
        free_resources_pq = []
        for free_resource in free_resources:
            heapq.heappush(
                free_resources_pq,
                (free_resource[metric_index], free_resource)
            )

        # Get allocation
        while len(required_resources_pq) > 0:
            # Get list, not tuple
            required_resource = heapq.heappop(required_resources_pq)[1]

            not_usable_free_resources = []
            is_allocated = False
            curr_free_resource = None
            while len(free_resources_pq) > 0:
                # Get list, not tuple
                curr_free_resource = heapq.heappop(free_resources_pq)[1]
                if curr_free_resource[1] >= required_resource[1] and \
                        curr_free_resource[2] >= required_resource[2] and \
                        curr_free_resource[3] >= required_resource[3]:
                    is_allocated = True
                    break
                else:
                    not_usable_free_resources.append(curr_free_resource)

            # Do allocation or return error
            if is_allocated:
                allocation_plan[required_resource[0]] = curr_free_resource[0]
                curr_free_resource[1] -= required_resource[1]
                curr_free_resource[2] -= required_resource[2]
                curr_free_resource[3] -= required_resource[3]
                heapq.heappush(
                    free_resources_pq,
                    (-curr_free_resource[metric_index], curr_free_resource)
                )
                for not_usable_free_resource in not_usable_free_resources:
                    heapq.heappush(
                        free_resources_pq,
                        (-not_usable_free_resource[metric_index], not_usable_free_resource)
                    )
            else:
                # add previous resources back, to do printing
                for not_usable_free_resource in not_usable_free_resources:
                    heapq.heappush(
                        free_resources_pq,
                        (-not_usable_free_resource[metric_index], not_usable_free_resource)
                    )
                heapq.heappush(
                    required_resources_pq,
                    (-required_resource[metric_index], required_resource)
                )

                logger.warning(allocation_plan)
                logger.warning(required_resources_pq)
                logger.warning(free_resources_pq)
                raise AllocationFailed("Unable to allocate, Abort")

        logger.info(required_resources)
        logger.info(free_resources)
        return allocation_plan

    @staticmethod
    def _get_single_metric_balanced_allocation_plan(
            allocation_details: dict, required_resources: list, free_resources: list) -> dict:
        # Init params
        allocation_plan = {}
        if 'metric' not in allocation_details or allocation_details['metric'].lower() not in METRIC_TO_INDEX:
            raise AllocationFailed("Invalid allocation parameter: metric")
        metric_index = METRIC_TO_INDEX[allocation_details['metric'].lower()]

        # Init reverse PQ
        required_resources_pq = []
        for required_resource in required_resources:
            print(required_resource)
            heapq.heappush(
                required_resources_pq,
                (-required_resource[metric_index], required_resource)
            )
        free_resources_pq = []
        for free_resource in free_resources:
            heapq.heappush(
                free_resources_pq,
                (-free_resource[metric_index], free_resource)
            )

        # Get allocation
        while len(required_resources_pq) > 0:
            # Get list, not tuple
            required_resource = heapq.heappop(required_resources_pq)[1]

            not_usable_free_resources = []
            is_allocated = False
            curr_free_resource = None
            while len(free_resources_pq) > 0:
                # Get list, not tuple
                curr_free_resource = heapq.heappop(free_resources_pq)[1]
                if curr_free_resource[1] >= required_resource[1] and \
                        curr_free_resource[2] >= required_resource[2] and \
                        curr_free_resource[3] >= required_resource[3]:
                    is_allocated = True
                    break
                else:
                    not_usable_free_resources.append(curr_free_resource)

            # Do allocation or return error
            if is_allocated:
                allocation_plan[required_resource[0]] = curr_free_resource[0]
                curr_free_resource[1] -= required_resource[1]
                curr_free_resource[2] -= required_resource[2]
                curr_free_resource[3] -= required_resource[3]
                heapq.heappush(
                    free_resources_pq,
                    (-curr_free_resource[metric_index], curr_free_resource)
                )
                for not_usable_free_resource in not_usable_free_resources:
                    heapq.heappush(
                        free_resources_pq,
                        (-not_usable_free_resource[metric_index], not_usable_free_resource)
                    )
            else:
                # add previous resources back, to do printing
                for not_usable_free_resource in not_usable_free_resources:
                    heapq.heappush(
                        free_resources_pq,
                        (-not_usable_free_resource[metric_index], not_usable_free_resource)
                    )
                heapq.heappush(
                    required_resources_pq,
                    (-required_resource[metric_index], required_resource)
                )

                logger.warning(allocation_plan)
                logger.warning(required_resources_pq)
                logger.warning(free_resources_pq)
                raise AllocationFailed("Unable to allocate, Abort")

        logger.info(required_resources)
        logger.info(free_resources)
        return allocation_plan

    def _get_required_resources(self, job_details: dict) -> list:
        # Load configs
        components_details = job_details['components']
        job_id = job_details['id']

        # Get required resources
        resources_list = []
        for component_type, component_details in components_details.items():
            component_id = component_details['id']
            component_num = component_details['num']
            required_cpu = component_details['resources']['cpu']
            required_memory = int(
                component_details['resources']['memory'].replace('m', ''))
            required_gpu = component_details['resources']['gpu']

            for i in range(component_num):
                resources_list.append(
                    [
                        f"{job_id}-{component_id}-{i}",
                        required_cpu,
                        required_memory,
                        required_gpu,
                    ]
                )
        return resources_list

    def _get_free_resources(self) -> list:
        # Load details
        nodes_details = get_nodes_details(
            redis=self._redis,
            cluster_name=self._cluster_name
        )

        # Get free resources
        free_resources_list = []
        for node_name, node_details in nodes_details.items():
            target_free_cpu = node_details['resources']['target_free_cpu']
            target_free_memory = node_details['resources']['target_free_memory']
            target_free_gpu = node_details['resources']['target_free_gpu']

            if node_details['state'] == 'Running':
                free_resources_list.append(
                    [
                        node_name,
                        target_free_cpu,
                        target_free_memory,
                        target_free_gpu
                    ]
                )
        return free_resources_list

    def remove_container(self, container_name: str):
        # Load details and vars
        nodes_details = get_nodes_details(
            redis=self._redis,
            cluster_name=self._cluster_name
        )
        admin_username = self._admin_username

        # Load command
        for node_name, node_details in nodes_details.items():
            node_hostname = node_details['hostname']

            command = REMOVE_CONTAINER_COMMAND.format(
                admin_username=admin_username,
                node_hostname=node_hostname,
                containers=container_name
            )

            completed_process = subprocess.run(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
            )
            if completed_process.returncode != 0:
                logger.error(f"No container {container_name} in {node_name}")
            logger.info(command)

    def start_container(self, container_name: str, node_name: str, job_details: dict):
        # Load details and vars
        cluster_name = self._cluster_name
        component_id_to_component_type = self._get_component_id_to_component_type(
            job_details=job_details)
        component_id = container_name.split('-')[-2]
        component_index = container_name.split('-')[-1]
        component_type = component_id_to_component_type[component_id]
        job_name = job_details['name']
        job_id = job_details['id']
        cluster_id = self._cluster_id
        admin_username = self._admin_username
        node_details = get_node_details(
            redis=self._redis,
            cluster_name=self._cluster_name,
            node_name=node_name
        )
        node_hostname = node_details['hostname']
        master_hostname = self._master_hostname

        # Parse environment parameters
        environment_parameters = \
            f"-e COMPONENT_TYPE={component_type} " \
            f"-e COMPONENT_ID={component_id} " \
            f"-e COMPONENT_INDEX={component_index} " \
            f"-e JOB_NAME={job_name} " \
            f"-e JOB_ID={job_id} " \
            f"-e CLUSTER_NAME={cluster_name} " \
            f"-e CLUSTER_ID={cluster_id} " \
            f"-e PYTHONUNBUFFERED=0"

        # Load command
        if job_details['components'][component_type]['resources']['gpu'] != 0:
            command = START_CONTAINER_WITH_GPU_COMMAND
        else:
            command = START_CONTAINER_COMMAND
        command = command.format(
            # cluster related
            admin_username=admin_username,
            master_hostname=master_hostname,
            node_hostname=node_hostname,
            fluentd_port=self._fluentd_port,

            # job related (user)
            cpu=job_details['components'][component_type]['resources']['cpu'],
            memory=job_details['components'][component_type]['resources']['memory'],
            gpu=job_details['components'][component_type]['resources']['gpu'],
            mount_target=job_details['components'][component_type]['mount']['target'],
            command=job_details['components'][component_type]['command'],
            image_name=job_details['components'][component_type]['image'],

            # job related (system)
            container_name=container_name,
            job_id=job_id,
            mount_source=f"~/.maro/clusters/{cluster_name}/data/",
            environment_parameters=environment_parameters
        )

        # Exec command
        logger.info(command)
        completed_process = subprocess.run(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
        )
        if completed_process.returncode != 0:
            raise AllocationFailed(completed_process.stderr)

    # Utils

    def _get_component_id_to_component_type(self, job_details: dict) -> dict:
        # Load variables
        components_details = job_details['components']

        # Get component_id_to_type
        component_id_to_type = {}
        for component_type, component_details in components_details.items():
            component_id_to_type[component_details['id']] = component_type

        return component_id_to_type


class KilledJobAgent(multiprocessing.Process):
    def __init__(self, cluster_name: str, redis_port: int, check_interval: int = 10):
        super().__init__()
        self._cluster_name = cluster_name
        self._redis = redis.Redis(
            host='localhost',
            port=redis_port,
            charset="utf-8", decode_responses=True
        )
        cluster_details = load_cluster_details(cluster_name=cluster_name)
        self._cluster_id = cluster_details['id']
        self._admin_username = cluster_details['user']['admin_username']

        self._check_interval = check_interval

        self._killed_jobs = []

    def run(self) -> None:
        while True:
            self._schedule_killed_jobs()
            time.sleep(self._check_interval)

    def _schedule_killed_jobs(self):
        self._killed_jobs = get_killed_jobs(
            redis=self._redis,
            cluster_name=self._cluster_name
        )

        for killed_job in self._killed_jobs:
            # Get details
            job_details = get_job_details(
                redis=self._redis,
                cluster_name=self._cluster_name,
                job_name=killed_job
            )
            job_id = job_details['id']

            # Kill job
            self._kill_job(job_id=job_id)

            # Remove killed job
            remove_killed_job(
                redis=self._redis,
                cluster_name=self._cluster_name,
                job_name=killed_job
            )

    def _kill_job(self, job_id: str):
        # Load details and vars
        nodes_details = get_nodes_details(
            redis=self._redis,
            cluster_name=self._cluster_name
        )
        admin_username = self._admin_username

        # Delete containers
        for node_name, node_details in nodes_details.items():
            # Load details
            container_details = node_details['containers']
            node_hostname = node_details['hostname']

            # Filter containers
            removable_containers = []
            for container_name in container_details:
                if container_name.startswith(job_id):
                    removable_containers.append(container_name)

            # Stop containers
            if len(removable_containers) > 0:
                command = STOP_CONTAINER_COMMAND.format(
                    admin_username=admin_username,
                    node_hostname=node_hostname,
                    containers=' '.join(removable_containers)
                )
                completed_process = subprocess.run(
                    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
                )
                if completed_process.returncode != 0:
                    logger.error(completed_process.stderr)
                logger.info(command)


class AllocationFailed(Exception):
    pass


class StartContainerFailed(Exception):
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(levelname)-7s] - %(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    with open(os.path.expanduser("~/.maro-local/agents/master_agent.config"), 'r') as fr:
        master_agent_config = json.load(fr)

    master_agent = MasterAgent(
        cluster_name=master_agent_config['cluster_name'],
        redis_port=master_agent_config['redis_port']
    )
    master_agent.start()
