# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import hashlib
import json
import os
import uuid

import yaml

"""Load from files"""


def load_cluster_details(cluster_name: str) -> dict:
    with open(os.path.expanduser(f"~/.maro/clusters/{cluster_name}/details.yml"), 'r') as fr:
        cluster_details = yaml.safe_load(fr)
    return cluster_details


def load_job_details(cluster_name: str, job_name: str) -> dict:
    with open(os.path.expanduser(f"~/.maro/clusters/{cluster_name}/jobs/{job_name}/details.yml"), 'r') as fr:
        job_details = yaml.safe_load(fr)
    return job_details


"""Generate ID"""


def generate_name_with_uuid(prefix: str, uuid_len: int = 16) -> str:
    postfix = uuid.uuid4().hex[:uuid_len]
    return f"{prefix}{postfix}"


"""Details from Redis"""


def get_node_details(redis, cluster_name: str, node_name: str) -> dict:
    return json.loads(
        redis.hget(
            f"{cluster_name}:node_details",
            node_name
        )
    )


def get_master_details(redis, cluster_name: str) -> dict:
    return json.loads(
        redis.get(
            f"{cluster_name}:master_details"
        )
    )


def get_nodes_details(redis, cluster_name: str) -> dict:
    nodes_details = redis.hgetall(
        f"{cluster_name}:node_details"
    )
    for node_name, node_details in nodes_details.items():
        nodes_details[node_name] = json.loads(node_details)
    return nodes_details


def set_node_details(redis, cluster_name: str, node_name: str, node_details: dict) -> None:
    redis.hset(
        f"{cluster_name}:node_details",
        node_name,
        json.dumps(node_details)
    )


def set_master_details(redis, cluster_name: str, master_details: dict) -> None:
    redis.set(
        f"{cluster_name}:master_details",
        json.dumps(master_details)
    )


def delete_node_details(redis, cluster_name: str, node_name: str):
    redis.hdel(
        f"{cluster_name}:node_details",
        node_name
    )


def get_job_details(redis, cluster_name: str, job_name: str) -> dict:
    return json.loads(
        redis.hget(
            f"{cluster_name}:job_details",
            job_name
        )
    )


def get_jobs_details(redis, cluster_name: str) -> dict:
    jobs_details = redis.hgetall(
        f"{cluster_name}:job_details",
    )
    for job_name, job_details in jobs_details.items():
        jobs_details[job_name] = json.loads(job_details)
    return jobs_details


def set_job_details(redis, cluster_name: str, job_name: str, job_details: dict) -> None:
    redis.hset(
        f"{cluster_name}:job_details",
        job_name,
        json.dumps(job_details)
    )


"""Hash related"""


def get_checksum(file_path: str, block_size=128):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size * md5.block_size), b''):
            md5.update(chunk)
    return md5.hexdigest()
