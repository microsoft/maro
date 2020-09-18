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
    return json.loads(
        redis.hget(
            f"{cluster_name}:job_details",
            job_name
        )
    )


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


"""Pending jobs"""


def get_pending_jobs(redis: Redis, cluster_name: str):
    return redis.lrange(
        f"{cluster_name}:pending_jobs",
        0,
        -1
    )


def remove_pending_job(redis: Redis, cluster_name: str, job_name: str):
    redis.lrem(
        f"{cluster_name}:pending_jobs",
        0,
        job_name
    )


"""Killed jobs"""


def get_killed_jobs(redis: Redis, cluster_name: str):
    return redis.lrange(
        f"{cluster_name}:killed_jobs",
        0,
        -1
    )


def remove_killed_job(redis: Redis, cluster_name: str, job_name: str):
    redis.lrem(
        f"{cluster_name}:killed_jobs",
        0,
        job_name
    )
