# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import base64
import json
import time

from maro.cli.utils.params import GlobalPaths
from maro.cli.utils.subprocess import SubProcess
from maro.utils.exception.cli_exception import CliException
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class GrassExecutor:
    def __init__(self, cluster_details: dict):
        self.cluster_details = cluster_details
        self.cluster_name = cluster_details['name']

        self.admin_username = self.cluster_details['user']['admin_username']

    def remote_build_image(self,
                           remote_context_path: str, remote_image_name: str):
        print("remote_build_image")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/build_image.py " \
                  f"{self.cluster_name} {remote_context_path} {remote_image_name}'"
        _ = SubProcess.run(command)

    def remote_clean(self, parallels: int):
        print("remote_clean")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/clean.py {self.cluster_name} {parallels}'"
        _ = SubProcess.run(command)

    def remote_get_checksum(self, file_path: str) -> str:
        print("remote_get_checksum")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/get_checksum.py {file_path}'"
        return_str = SubProcess.run(command)
        return return_str

    def remote_get_jobs_details(self):
        print("remote_get_jobs_details")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/get_jobs_details.py {self.cluster_name}'"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_master_details(self):
        print("remote_get_master_details")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/get_master_details.py {self.cluster_name}'"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_node_details(self, node_name: str):
        print("remote_get_node_details")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/get_node_details.py {self.cluster_name} {node_name}'"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_nodes_details(self):
        print("remote_get_nodes_details")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/get_nodes_details.py {self.cluster_name}'"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_public_key(self, node_ip_address: str):
        print("remote_get_public_key")
        command = f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/get_public_key.py'"
        print(command)
        return_str = SubProcess.run(command).strip('\n')
        logger.debug(return_str)
        return return_str

    def remote_init_master(self):
        print("remote_init_master")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/init_master.py {self.cluster_name}'"
        SubProcess.interactive_run(command)

    def remote_init_node(self, node_name: str, node_ip_address: str):
        print("remote_init_node")
        command = f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} " \
                  f"'python3 ~/init_node.py {self.cluster_name} {node_name}'"
        SubProcess.interactive_run(command)

    def remote_load_images(self, node_name: str, parallels: int, node_ip_address: str):
        print("remote_load_images")
        command = f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/load_images.py " \
                  f"{self.cluster_name} {node_name} {parallels}'"
        SubProcess.interactive_run(command)

    def remote_load_master_agent_service(self):
        print("remote_load_master_agent_service")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/load_master_agent_service.py {self.cluster_name}'"
        _ = SubProcess.run(command)

    def remote_load_node_agent_service(self, node_name: str, node_ip_address: str):
        print("remote_load_node_agent_service")
        command = f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/load_node_agent_service.py " \
                  f"{self.cluster_name} {node_name}'"
        _ = SubProcess.run(command)

    def remote_create_pending_job_ticket(self, job_name: str):
        print("remote_create_pending_job_ticket")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/create_pending_job_ticket.py " \
                  f"{self.cluster_name} {job_name}'"
        _ = SubProcess.run(command)

    def remote_create_job_details(self, job_name: str):
        print("remote_create_job_details")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/create_job_details.py " \
                  f"{self.cluster_name} {job_name}'"
        _ = SubProcess.run(command)

    def remote_create_killed_job_ticket(self, job_name: str):
        print("remote_create_killed_job_ticket")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/create_killed_job_ticket.py " \
                  f"{self.cluster_name} {job_name}'"
        _ = SubProcess.run(command)

    def remote_delete_pending_job_ticket(self, job_name: str):
        print("remote_delete_pending_job_ticket")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/delete_pending_job_ticket.py " \
                  f"{self.cluster_name} {job_name}'"
        _ = SubProcess.run(command)

    def remote_set_master_details(self, master_details: dict):
        print("remote_set_master_details")
        master_details_b64 = base64.b64encode(json.dumps(master_details).encode("utf8")).decode('utf8')
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/set_master_details.py " \
                  f"{self.cluster_name} {master_details_b64}'"
        _ = SubProcess.run(command)

    def remote_set_node_details(self, node_name: str, node_details: dict):
        print("remote_set_node_details")
        node_details_b64 = base64.b64encode(json.dumps(node_details).encode("utf8")).decode('utf8')
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/set_node_details.py " \
                  f"{self.cluster_name} {node_name} {node_details_b64}'"
        _ = SubProcess.run(command)

    def remote_update_image_files_details(self):
        print("remote_update_image_files_details")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/update_image_files_details.py " \
                  f"{self.cluster_name}'"
        _ = SubProcess.run(command)

    def remote_update_node_status(self, node_name: str, action: str):
        print("remote_update_node_status")
        command = f"ssh -o StrictHostKeyChecking=no " \
                  f"{self.admin_username}@{self.cluster_details['master']['public_ip_address']} " \
                  f"'python3 {GlobalPaths.MARO_GRASS_LIB}/scripts/update_node_status.py " \
                  f"{self.cluster_name} {node_name} {action}'"
        _ = SubProcess.run(command)

    def test_connection(self, node_ip_address: str):
        print("test_connection")
        command = f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} " \
                  f"echo 'Connection established'"
        _ = SubProcess.run(command)

    def retry_until_connected(self, node_ip_address: str) -> bool:
        print("retry_until_connected")
        remain_retries = 10
        while remain_retries > 0:
            try:
                self.test_connection(node_ip_address)
                return True
            except CliException:
                remain_retries -= 1
                logger.debug(
                    f"Unable to connect to {node_ip_address}, remains {remain_retries} retries")
                time.sleep(10)
                continue

        raise CliException(f"Unable to connect to {node_ip_address}")

    def remote_interactive_connect(self, node_ip_address: str):
        print("remote_interactive_connect")
        command = f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} " \
                  f""
        SubProcess.interactive_run(command)

    @staticmethod
    # Create a new user account on target OS.
    def remote_add_user_to_node(admin_username: str, maro_user: str, node_ip_address: str, pubkey: str):
        print("remote_add_user_to_node")
        # The admin_user is an already exist account which has privileges to create new account on target OS.
        command = f"ssh " \
                  f"{admin_username}@{node_ip_address} " \
                  f"'sudo python3 ~/create_user.py " \
                  f"{maro_user} " \
                  f"\"{pubkey}\"'"
        _ = SubProcess.run(command)

    # Delete maro cluster user account on target OS.
    @staticmethod
    def remote_delete_user_from_node(admin_username: str, delete_user: str, node_ip_address: str):
        print("remote_delete_user_from_node")
        # The admin_user is an already exist account which has privileges to create new account on target OS.
        command = f"ssh " \
                  f"{admin_username}@{node_ip_address} " \
                  f"'sudo python3 ~/delete_user.py " \
                  f"{delete_user}'"
        _ = SubProcess.run(command)
