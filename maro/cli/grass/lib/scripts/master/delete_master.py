# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


""" Delete Master.

[WARNING] This script is a standalone script, which cannot use the ./utils tools.

Do the following works:
- Stop master services (agent and api-server).
- Remove Redis and Fluentd containers.
"""

import os
import subprocess
import sys

import yaml

# Commands

STOP_MASTER_AGENT_SERVICE_COMMAND = """systemctl --user stop maro-master-agent.service"""

STOP_MASTER_API_SERVER_SERVICE_COMMAND = """systemctl --user stop maro-master-api-server.service"""

REMOVE_CONTAINERS = """sudo docker rm -f maro-fluentd-{cluster_id} maro-redis-{cluster_id}"""


# Master Deleter.

class MasterDeleter:
    def __init__(self, local_cluster_details: dict):
        self._local_cluster_details = local_cluster_details

    @staticmethod
    def stop_master_agent_service():
        Subprocess.run(command=STOP_MASTER_AGENT_SERVICE_COMMAND)
        os.remove(os.path.expanduser("~/.config/systemd/user/maro-master-agent.service"))

    @staticmethod
    def stop_master_api_server_service():
        Subprocess.run(command=STOP_MASTER_API_SERVER_SERVICE_COMMAND)
        os.remove(os.path.expanduser("~/.config/systemd/user/maro-master-api-server.service"))

    def remove_containers(self):
        command = REMOVE_CONTAINERS.format(cluster_id=self._local_cluster_details["id"])
        Subprocess.run(command=command)


# Utils Classes.


class Paths:
    MARO_LOCAL = "~/.maro-local"
    ABS_MARO_LOCAL = os.path.expanduser(MARO_LOCAL)


class DetailsReader:
    @staticmethod
    def load_local_cluster_details() -> dict:
        with open(file=f"{Paths.ABS_MARO_LOCAL}/cluster/cluster_details.yml", mode="r") as fr:
            cluster_details = yaml.safe_load(stream=fr)
        return cluster_details


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


if __name__ == "__main__":
    master_releaser = MasterDeleter(local_cluster_details=DetailsReader.load_local_cluster_details())
    master_releaser.stop_master_agent_service()
    master_releaser.stop_master_api_server_service()
    master_releaser.remove_containers()
