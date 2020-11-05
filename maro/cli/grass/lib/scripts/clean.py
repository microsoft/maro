# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import subprocess
import sys
from multiprocessing.pool import ThreadPool

from redis import Redis

from .utils import get_nodes_details, load_cluster_details

LIST_CONTAINERS_COMMAND = """\
ssh -o StrictHostKeyChecking=no {admin_username}@{node_hostname} \
docker ps -aq\
"""

STOP_CONTAINERS_COMMAND = """\
ssh -o StrictHostKeyChecking=no {admin_username}@{node_hostname} \
docker stop {containers}\
"""

REMOVE_CONTAINERS_COMMAND = """\
ssh -o StrictHostKeyChecking=no {admin_username}@{node_hostname} \
docker rm -f {containers}\
"""


def _clean_cluster_containers(cluster_name: str, parallels: int):
    # Load details
    cluster_details = load_cluster_details(cluster_name=cluster_name)
    admin_username = cluster_details['user']['admin_username']
    master_hostname = cluster_details['master']['hostname']
    redis_port = cluster_details['master']['redis']['port']
    redis = Redis(
        host=master_hostname,
        port=redis_port,
        charset="utf-8", decode_responses=True
    )
    nodes_details = get_nodes_details(
        redis,
        cluster_name=cluster_name
    )

    # Parallel clean
    with ThreadPool(parallels) as pool:
        params = [
            [
                admin_username,
                node_details['hostname']
            ]
            for _, node_details in nodes_details.items()
        ]
        pool.starmap(
            _clean_node_containers,
            params
        )


def _clean_node_containers(admin_username: str, node_hostname: str):
    # Get containers
    command = LIST_CONTAINERS_COMMAND.format(
        admin_username=admin_username,
        node_hostname=node_hostname
    )
    completed_process = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
    )
    if completed_process.returncode != 0:
        raise Exception(completed_process.stderr)
    return_str = completed_process.stdout.strip('\n')
    if return_str == '':
        return
    containers = return_str.split('\n')

    # Stop containers with SIGTERM with default(10s) grace period
    command = STOP_CONTAINERS_COMMAND.format(
        admin_username=admin_username,
        node_hostname=node_hostname,
        containers=' '.join(containers)
    )
    completed_process = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
    )
    if completed_process.returncode != 0:
        sys.stderr.write(completed_process.stderr + '\n')
    sys.stdout.write(command + '\n')

    # Remove containers
    command = REMOVE_CONTAINERS_COMMAND.format(
        admin_username=admin_username,
        node_hostname=node_hostname,
        containers=' '.join(containers)
    )
    completed_process = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
    )
    if completed_process.returncode != 0:
        sys.stderr.write(completed_process.stderr + '\n')
    sys.stdout.write(command + '\n')


if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_name')
    parser.add_argument('parallels', type=int)
    args = parser.parse_args()

    # clean all docker
    _clean_cluster_containers(cluster_name=args.cluster_name, parallels=args.parallels)
