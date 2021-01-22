# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Activate "leave cluster" operation.

[WARNING] This script is a standalone script, which cannot use the ./utils tools.

The script will do the following jobs in this VM:
- Delete node at master-api-server (the actual delete action will be executed at master-api-server, see
    lib/services/master_api_server/blueprints/nodes.py for reference.
"""

import os
import subprocess
import sys

import redis
import yaml


class Paths:
    MARO_LOCAL = "~/.maro-local"
    ABS_MARO_LOCAL = os.path.expanduser(MARO_LOCAL)


class DetailsReader:
    @staticmethod
    def load_local_cluster_details() -> dict:
        with open(file=f"{Paths.ABS_MARO_LOCAL}/cluster/cluster_details.yml", mode="r") as fr:
            cluster_details = yaml.safe_load(stream=fr)
        return cluster_details

    @staticmethod
    def load_local_node_details() -> dict:
        with open(file=f"{Paths.ABS_MARO_LOCAL}/cluster/node_details.yml", mode="r") as fr:
            node_details = yaml.safe_load(stream=fr)
        return node_details


class RedisController:
    def __init__(self, host: str, port: int):
        self._redis = redis.Redis(host=host, port=port, encoding="utf-8", decode_responses=True)

    """Node Details Related."""

    def delete_node_details(self, node_name: str) -> None:
        self._redis.hdel(
            "name_to_node_details",
            node_name
        )


class Subprocess:
    @staticmethod
    def run(command: str, timeout: int = None) -> None:
        """Run one-time command with subprocess.run().

        Args:
            command (str): command to be executed.
            timeout (int): timeout in seconds.

        Returns:
            str: return stdout of the command.
        """
        # TODO: Windows node
        completed_process = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout
        )
        if completed_process.returncode != 0:
            raise Exception(completed_process.stderr)
        sys.stderr.write(completed_process.stderr)


if __name__ == '__main__':
    local_cluster_details = DetailsReader.load_local_cluster_details()
    local_node_details = DetailsReader.load_local_node_details()

    command = f"python3 {Paths.MARO_LOCAL}/scripts/leave_cluster.py"
    Subprocess.run(command=command)

    redis_controller = RedisController(
        host=local_cluster_details["master"]["private_ip_address"],
        port=local_cluster_details["master"]["redis"]["port"]
    )
    redis_controller.delete_node_details(node_name=local_node_details["name"])
