# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import logging
import time
from subprocess import TimeoutExpired

from .exception import CommandExecutionError, ConnectionFailed
from .params import Paths
from .subprocess import Subprocess

logger = logging.getLogger(__name__)


class ConnectionTester:
    """Tester class for connection.
    """

    @staticmethod
    def test_ssh_default_port_connection(node_username: str, node_hostname: str, node_ssh_port: int, cluster_name: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no "
            f"-i {Paths.MARO_LOCAL}/cluster/{cluster_name}/master_to_node_openssh_private_key "
            f"-p {node_ssh_port} {node_username}@{node_hostname} "
            "echo 'Connection established'"
        )
        _ = Subprocess.run(command=command, timeout=5)

    @staticmethod
    def retry_connection(node_username: str, node_hostname: str, node_ssh_port: int, cluster_name: str) -> bool:
        remain_retries = 20
        while remain_retries > 0:
            try:
                ConnectionTester.test_ssh_default_port_connection(
                    node_ssh_port=node_ssh_port,
                    node_username=node_username,
                    node_hostname=node_hostname,
                    cluster_name=cluster_name
                )
                return True
            except (CommandExecutionError, TimeoutExpired):
                remain_retries -= 1
                logger.debug(
                    f"Unable to connect to {node_hostname} with port {node_ssh_port}, "
                    f"remains {remain_retries} retries"
                )
                time.sleep(5)
        raise ConnectionFailed(f"Unable to connect to {node_hostname} with port {node_ssh_port}")
