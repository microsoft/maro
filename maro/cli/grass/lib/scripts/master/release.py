# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import subprocess
import sys

# Commands

STOP_MASTER_AGENT_SERVICE_COMMAND = """systemctl --user stop maro-master-agent.service"""

STOP_MASTER_API_SERVER_SERVICE_COMMAND = """systemctl --user stop maro-master-api-server.service"""

REMOVE_CONTAINERS = """sudo docker rm -f maro-fluentd maro-redis"""


class Paths:
    MARO_SHARED = "~/.maro-shared"
    MARO_LOCAL = "~/.maro-local"

    ABS_MARO_SHARED = os.path.expanduser(MARO_SHARED)
    ABS_MARO_LOCAL = os.path.expanduser(MARO_LOCAL)


class MasterReleaser:
    @staticmethod
    def stop_master_agent_service():
        Subprocess.run(command=STOP_MASTER_AGENT_SERVICE_COMMAND)
        os.remove(os.path.expanduser("~/.config/systemd/user/maro-master-agent.service"))
        os.remove(f"{Paths.ABS_MARO_LOCAL}/services/maro-master-agent.config")

    @staticmethod
    def stop_master_api_server_service():
        Subprocess.run(command=STOP_MASTER_API_SERVER_SERVICE_COMMAND)
        os.remove(os.path.expanduser("~/.config/systemd/user/maro-master-api-server.service"))

    @staticmethod
    def remove_containers():
        Subprocess.run(command=REMOVE_CONTAINERS)


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
    master_releaser = MasterReleaser()
    master_releaser.stop_master_agent_service()
    master_releaser.stop_master_api_server_service()
    master_releaser.remove_containers()
