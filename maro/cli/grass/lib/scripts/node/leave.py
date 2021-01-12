# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import subprocess
import sys

import requests
import yaml

# Commands

UNMOUNT_FOLDER_COMMAND = """sudo umount -f {maro_shared_path}"""

STOP_NODE_AGENT_SERVICE_COMMAND = """systemctl --user stop maro-node-agent.service"""

STOP_NODE_API_SERVER_SERVICE_COMMAND = """systemctl --user stop maro-node-api-server.service"""


class Paths:
    MARO_SHARED = "~/.maro-shared"
    MARO_LOCAL = "~/.maro-local"

    ABS_MARO_SHARED = os.path.expanduser(MARO_SHARED)
    ABS_MARO_LOCAL = os.path.expanduser(MARO_LOCAL)


class NodeLeaver:
    def __init__(self):
        local_cluster_details = DetailsReader.load_local_cluster_details()
        local_node_details = DetailsReader.load_local_node_details()

        master_api_client = MasterApiClientV1(
            master_hostname=local_cluster_details["master"]["hostname"],
            api_server_port=local_cluster_details["connection"]["api_server"]["port"]
        )
        master_api_client.delete_node(node_name=local_node_details["name"])

    @staticmethod
    def umount_maro_share():
        command = UNMOUNT_FOLDER_COMMAND.format(maro_shared_path=Paths.ABS_MARO_SHARED)
        Subprocess.run(command=command)

    @staticmethod
    def stop_node_agent_service():
        Subprocess.run(command=STOP_NODE_AGENT_SERVICE_COMMAND)
        os.remove(os.path.expanduser("~/.config/systemd/user/maro-node-agent.service"))
        os.remove(f"{Paths.ABS_MARO_LOCAL}/services/maro-node-agent.config")

    @staticmethod
    def stop_node_api_server_service():
        Subprocess.run(command=STOP_NODE_API_SERVER_SERVICE_COMMAND)
        os.remove(os.path.expanduser("~/.config/systemd/user/maro-node-api-server.service"))


class MasterApiClientV1:
    def __init__(self, master_hostname: str, api_server_port: int):
        print(master_hostname, api_server_port)
        self.master_api_server_url_prefix = f"http://{master_hostname}:{api_server_port}/v1"

    # Node related.

    def delete_node(self, node_name: str) -> dict:
        response = requests.delete(url=f"{self.master_api_server_url_prefix}/nodes/{node_name}")
        return response.json()


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
