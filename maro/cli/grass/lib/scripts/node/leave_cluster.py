# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Leave the current VM off the MARO Cluster.

[WARNING] This script is a standalone script, which cannot use the ./utils tools.

The script will do the following jobs in this VM:
- Unmount samba server.
- Stop MARO Node services: maro-node-agent, maro-node-api-server.
"""

import logging
import os
import subprocess
import sys

import yaml

# Commands

UNMOUNT_FOLDER_COMMAND = """sudo umount -f {maro_shared_path}"""

STOP_NODE_AGENT_SERVICE_COMMAND = """systemctl --user stop maro-node-agent.service"""

STOP_NODE_API_SERVER_SERVICE_COMMAND = """systemctl --user stop maro-node-api-server.service"""


class Paths:
    MARO_SHARED = "~/.maro-shared"
    ABS_MARO_SHARED = os.path.expanduser(MARO_SHARED)

    MARO_LOCAL = "~/.maro-local"
    ABS_MARO_LOCAL = os.path.expanduser(MARO_LOCAL)


class NodeLeaver:
    @staticmethod
    def umount_maro_share():
        command = UNMOUNT_FOLDER_COMMAND.format(maro_shared_path=Paths.ABS_MARO_SHARED)
        Subprocess.run(command=command)

    @staticmethod
    def stop_node_agent_service():
        Subprocess.run(command=STOP_NODE_AGENT_SERVICE_COMMAND)
        try:
            os.remove(os.path.expanduser("~/.config/systemd/user/maro-node-agent.service"))
        except FileNotFoundError:
            logging.warning("maro-node-agent.service is not found")

    @staticmethod
    def stop_node_api_server_service():
        Subprocess.run(command=STOP_NODE_API_SERVER_SERVICE_COMMAND)
        try:
            os.remove(os.path.expanduser("~/.config/systemd/user/maro-node-api-server.service"))
        except FileNotFoundError:
            logging.warning("maro-node-api-server.service is not found")


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
    node_leaver = NodeLeaver()
    node_leaver.stop_node_agent_service()
    node_leaver.stop_node_api_server_service()
    node_leaver.umount_maro_share()
