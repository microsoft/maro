# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import base64
import json
import time

from maro.cli.utils.params import GlobalPaths
from maro.cli.utils.subprocess import SubProcess
from maro.utils.exception.cli_exception import CliError, ClusterInternalError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class GrassExecutor:
    def __init__(self, cluster_details: dict):
        self.cluster_details = cluster_details
        self.cluster_name = cluster_details["name"]

        self.admin_username = self.cluster_details["user"]['admin_username']

    def remote_build_image(self, remote_context_path: str, remote_image_name: str):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.build_image "
            f"{self.cluster_name} {remote_context_path} {remote_image_name}'"
        )
        _ = SubProcess.run(command)

    def remote_clean(self, parallels: int):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.clean {self.cluster_name} {parallels}'"
        )
        _ = SubProcess.run(command)

    def remote_get_checksum(self, file_path: str) -> str:
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.get_checksum {file_path}'"
        )
        return_str = SubProcess.run(command)
        return return_str

    def remote_get_jobs_details(self):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.get_jobs_details {self.cluster_name}'"
        )
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_master_details(self):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.get_master_details {self.cluster_name}'"
        )
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_node_details(self, node_name: str):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.get_node_details {self.cluster_name} {node_name}'"
        )
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_nodes_details(self):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.get_nodes_details {self.cluster_name}'"
        )
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_containers_details(self):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.get_containers_details {self.cluster_name}'"
        )
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_public_key(self, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.get_public_key'"
        )
        return_str = SubProcess.run(command).strip("\n")
        logger.debug(return_str)
        return return_str

    def remote_init_build_node_image_vm(self, vm_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{vm_ip_address} "
            "'python3 ~/init_build_node_image_vm.py'"
        )
        SubProcess.interactive_run(command)

    def remote_init_master(self):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.init_master {self.cluster_name}'"
        )
        SubProcess.interactive_run(command)

    def remote_init_node(self, node_name: str, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} "
            f"'python3 ~/init_node.py {self.cluster_name} {node_name}'"
        )
        SubProcess.interactive_run(command)

    def remote_mkdir(self, node_ip_address: str, path: str):
        command = f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} 'mkdir -p {path}'"
        SubProcess.run(command)

    def remote_load_images(self, node_name: str, parallels: int, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.load_images "
            f"{self.cluster_name} {node_name} {parallels}'"
        )
        SubProcess.interactive_run(command)

    def remote_load_master_agent_service(self):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.load_master_agent_service {self.cluster_name}'"
        )
        _ = SubProcess.run(command)

    def remote_load_node_agent_service(self, node_name: str, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.load_node_agent_service "
            f"{self.cluster_name} {node_name}'"
        )
        _ = SubProcess.run(command)

    def remote_create_pending_job_ticket(self, job_name: str):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.create_pending_job_ticket "
            f"{self.cluster_name} {job_name}'"
        )
        _ = SubProcess.run(command)

    def remote_create_job_details(self, job_name: str):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.create_job_details "
            f"{self.cluster_name} {job_name}'"
        )
        _ = SubProcess.run(command)

    def remote_create_killed_job_ticket(self, job_name: str):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.create_killed_job_ticket "
            f"{self.cluster_name} {job_name}'"
        )
        _ = SubProcess.run(command)

    def remote_delete_pending_job_ticket(self, job_name: str):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.delete_pending_job_ticket "
            f"{self.cluster_name} {job_name}'"
        )
        _ = SubProcess.run(command)

    def remote_set_master_details(self, master_details: dict):
        master_details_b64 = base64.b64encode(json.dumps(master_details).encode("utf8")).decode('utf8')
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.set_master_details "
            f"{self.cluster_name} {master_details_b64}'"
        )
        _ = SubProcess.run(command)

    def remote_set_node_details(self, node_name: str, node_details: dict):
        node_details_b64 = base64.b64encode(json.dumps(node_details).encode("utf8")).decode('utf8')
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.set_node_details "
            f"{self.cluster_name} {node_name} {node_details_b64}'"
        )
        _ = SubProcess.run(command)

    def remote_update_image_files_details(self):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.update_image_files_details "
            f"{self.cluster_name}'"
        )
        _ = SubProcess.run(command)

    def remote_update_node_status(self, node_name: str, action: str):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.update_node_status "
            f"{self.cluster_name} {node_name} {action}'"
        )
        _ = SubProcess.run(command)

    def test_connection(self, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} "
            "echo 'Connection established'"
        )
        _ = SubProcess.run(command)

    def retry_until_connected(self, node_ip_address: str) -> bool:
        remain_retries = 10
        while remain_retries > 0:
            try:
                self.test_connection(node_ip_address)
                return True
            except CliError:
                remain_retries -= 1
                logger.debug(f"Unable to connect to {node_ip_address}, remains {remain_retries} retries.")
                time.sleep(10)
                continue
        raise ClusterInternalError(f"Unable to connect to {node_ip_address}.")
