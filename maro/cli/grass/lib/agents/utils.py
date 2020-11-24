# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import os

import yaml
from redis import Redis

"""Load from files"""


def load_cluster_details(cluster_name: str) -> dict:
    with open(os.path.expanduser(f"~/.maro/clusters/{cluster_name}/details.yml"), 'r') as fr:
        cluster_details = yaml.safe_load(fr)
    return cluster_details


def load_job_details(cluster_name: str, job_name: str) -> dict:
    with open(os.path.expanduser(f"~/.maro/clusters/{cluster_name}/jobs/{job_name}/details.yml"), 'r') as fr:
        job_details = yaml.safe_load(fr)
    return job_details


"""Node details"""


def get_node_details(redis: Redis, cluster_name: str, node_name: str) -> dict:
    return json.loads(
        redis.hget(
            f"{cluster_name}:node_details",
            node_name
        )
    )


def get_nodes_details(redis: Redis, cluster_name: str) -> dict:
    nodes_details = redis.hgetall(
        f"{cluster_name}:node_details"
    )
    for node_name, node_details in nodes_details.items():
        nodes_details[node_name] = json.loads(node_details)
    return nodes_details


def set_node_details(redis: Redis, cluster_name: str, node_name: str, node_details: dict) -> None:
    redis.hset(
        f"{cluster_name}:node_details",
        node_name,
        json.dumps(node_details)
    )


"""Job details"""


def get_job_details(redis: Redis, cluster_name: str, job_name: str) -> dict:
    return_str = redis.hget(
        f"{cluster_name}:job_details",
        job_name
    )
    return json.loads(return_str) if return_str is not None else None


def get_jobs_details(redis: Redis, cluster_name: str) -> dict:
    jobs_details = redis.hgetall(
        f"{cluster_name}:job_details",
    )
    for job_name, job_details in jobs_details.items():
        jobs_details[job_name] = json.loads(job_details)
    return jobs_details


def set_job_details(redis: Redis, cluster_name: str, job_name: str, job_details: dict) -> None:
    redis.hset(
        f"{cluster_name}:job_details",
        job_name,
        json.dumps(job_details)
    )


"""Containers details"""


def get_containers_details(redis: Redis, cluster_name: str) -> dict:
    containers_details = redis.hgetall(
        f"{cluster_name}:container_details",
    )
    for container_name, container_details in containers_details.items():
        containers_details[container_name] = json.loads(container_details)
    return containers_details


def set_containers_details(redis: Redis, cluster_name: str, containers_details: dict) -> None:
    redis.delete(f"{cluster_name}:container_details")
    if len(containers_details) == 0:
        return
    else:
        for container_name, container_details in containers_details.items():
            containers_details[container_name] = json.dumps(container_details)
        redis.hmset(
            f"{cluster_name}:container_details",
            containers_details
        )


def set_container_details(redis: Redis, cluster_name: str, container_name: str, container_details: dict) -> None:
    redis.hset(
        f"{cluster_name}:container_details",
        container_name,
        container_details
    )


"""Pending job ticket"""


def get_pending_job_tickets(redis: Redis, cluster_name: str):
    return redis.lrange(
        f"{cluster_name}:pending_job_tickets",
        0,
        -1
    )


def remove_pending_job_ticket(redis: Redis, cluster_name: str, job_name: str):
    redis.lrem(
        f"{cluster_name}:pending_job_tickets",
        0,
        job_name
    )


"""Killed job ticket"""


def get_killed_job_tickets(redis: Redis, cluster_name: str):
    return redis.lrange(
        f"{cluster_name}:killed_job_tickets",
        0,
        -1
    )


def remove_killed_job_ticket(redis: Redis, cluster_name: str, job_name: str):
    redis.lrem(
        f"{cluster_name}:killed_job_tickets",
        0,
        job_name
    )


"""Fault tolerance related"""


def get_rejoin_component_name_to_container_name(redis: Redis, job_id: str) -> dict:
    return redis.hgetall(
        f"job:{job_id}:rejoin_component_name_to_container_name"
    )


def get_rejoin_container_name_to_component_name(redis: Redis, job_id: str) -> dict:
    component_name_to_container_name = get_rejoin_component_name_to_container_name(
        redis=redis,
        job_id=job_id
    )
    return {v: k for k, v in component_name_to_container_name.items()}


def delete_rejoin_container_name_to_component_name(redis: Redis, job_id: str) -> None:
    redis.delete(
        f"job:{job_id}:rejoin_component_name_to_container_name"
    )


def get_job_runtime_details(redis: Redis, job_id: str) -> dict:
    return redis.hgetall(
        f"job:{job_id}:runtime_details"
    )


def get_rejoin_component_restart_times(redis, job_id: str, component_id: str) -> int:
    restart_times = redis.hget(
        f"job:{job_id}:component_id_to_restart_times",
        component_id
    )
    return 0 if restart_times is None else int(restart_times)


def incr_rejoin_component_restart_times(redis, job_id: str, component_id: str) -> None:
    redis.hincrby(
        f"job:{job_id}:component_id_to_restart_times",
        component_id,
        1
    )
