# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import hashlib
import json
import os
import time
from subprocess import TimeoutExpired

import yaml

from maro.cli.grass.utils.docker_controller import DockerController
from maro.cli.grass.utils.file_synchronizer import FileSynchronizer
from maro.cli.grass.utils.master_api_client import MasterApiClientV1
from maro.cli.utils.deployment_validator import DeploymentValidator
from maro.cli.utils.name_creator import NameCreator
from maro.cli.utils.params import GlobalPaths
from maro.cli.utils.subprocess import SubProcess
from maro.utils.exception.cli_exception import (
    BadRequestError, CliError, ClusterInternalError, CommandExecutionError, FileOperationError
)
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class GrassExecutor:
    def __init__(self, cluster_details: dict):
        self.cluster_details = cluster_details

        # Upper configs
        self.cluster_name = self.cluster_details["name"]
        self.cluster_id = self.cluster_details["id"]

        # User configs
        self.admin_username = self.cluster_details["user"]["admin_username"]

        # Master configs (may be dynamically create)
        self.master_public_ip_address = self.cluster_details["master"]["public_ip_address"]
        self.master_api_client = MasterApiClientV1(
            master_ip_address=self.master_public_ip_address,
            api_server_port=self.cluster_details["connection"]["api_server"]["port"]
        )

        # Connection configs
        self.ssh_port = self.cluster_details["connection"]["ssh"]["port"]
        self.api_server_port = self.cluster_details["connection"]["api_server"]["port"]

    # maro grass node

    def list_node(self):
        # Get nodes details
        nodes_details = self.master_api_client.list_nodes()

        # Print details
        logger.info(
            json.dumps(
                nodes_details,
                indent=4, sort_keys=True
            )
        )

    # maro grass image

    def push_image(self, image_name: str, image_path: str, remote_context_path: str, remote_image_name: str):
        # Push image TODO: design a new paradigm for remote build
        if image_name or image_path:
            if image_name:
                # Push image from local docker client.
                new_file_name = NameCreator.get_valid_file_name(image_name)
                abs_image_path = f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/image_files/{new_file_name}"
                DockerController.save_image(
                    image_name=image_name,
                    abs_export_path=abs_image_path
                )
            else:
                # Push image from local image file.
                file_name = os.path.basename(image_path)
                new_file_name = NameCreator.get_valid_file_name(file_name)
                abs_image_path = f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/image_files/{new_file_name}"
                FileSynchronizer.copy_and_rename(
                    source_path=image_path,
                    target_dir=f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/image_files",
                    new_name=new_file_name
                )
            remote_image_file_details = self.master_api_client.get_image_file(image_file_name=new_file_name)
            local_md5_checksum = self._get_md5_checksum(path=abs_image_path)
            if (
                "md5_checksum" in remote_image_file_details
                and remote_image_file_details["md5_checksum"] == local_md5_checksum
            ):
                logger.info_green(f"The image file '{new_file_name}' already exists")
                return
            FileSynchronizer.copy_files_to_node(
                local_path=abs_image_path,
                remote_dir=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/image_files",
                admin_username=self.admin_username,
                node_ip_address=self.master_public_ip_address,
                ssh_port=self.ssh_port
            )
            self.master_api_client.create_image_file(
                image_file_details={
                    "name": new_file_name,
                    "md5_checksum": local_md5_checksum
                }
            )
            logger.info_green(f"Image {image_name} is loaded")
        else:
            raise BadRequestError("Invalid arguments")

    # maro grass data

    def push_data(self, local_path: str, remote_path: str):
        if not remote_path.startswith("/"):
            raise FileOperationError(f"Invalid remote path: {remote_path}\nShould be started with '/'")
        FileSynchronizer.copy_files_to_node(
            local_path=local_path,
            remote_dir=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/data{remote_path}",
            admin_username=self.admin_username,
            node_ip_address=self.master_public_ip_address,
            ssh_port=self.ssh_port
        )

    def pull_data(self, local_path: str, remote_path: str):
        if not remote_path.startswith("/"):
            raise FileOperationError(f"Invalid remote path: {remote_path}\nShould be started with '/'")
        FileSynchronizer.copy_files_from_node(
            local_dir=local_path,
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/data{remote_path}",
            admin_username=self.admin_username,
            node_ip_address=self.master_public_ip_address,
            ssh_port=self.ssh_port
        )

    # maro grass job

    def start_job(self, deployment_path: str):
        # Load start_job_deployment
        with open(deployment_path, "r") as fr:
            start_job_deployment = yaml.safe_load(fr)

        self._start_job(start_job_deployment=start_job_deployment)

    def _start_job(self, start_job_deployment: dict):
        # Standardize start_job_deployment
        job_details = self._standardize_job_details(start_job_deployment=start_job_deployment)

        # Create job
        logger.info(f"Sending job ticket '{start_job_deployment['name']}'")
        self.master_api_client.create_job(job_details=job_details)
        logger.info_green(f"Job ticket '{job_details['name']}' is sent")

    def stop_job(self, job_name: str):
        # Delete job
        self.master_api_client.delete_job(job_name=job_name)

    def list_job(self):
        # Get jobs details
        jobs_details = self.master_api_client.list_jobs()

        # Print details
        logger.info(
            json.dumps(
                jobs_details,
                indent=4, sort_keys=True
            )
        )

    def get_job_logs(self, job_name: str, export_dir: str = "./"):
        # Load details
        job_details = self.master_api_client.get_job(job_name=job_name)

        # Copy logs from master
        try:
            FileSynchronizer.copy_files_from_node(
                local_dir=export_dir,
                remote_path=f"~/.maro/logs/{job_details['id']}",
                admin_username=self.admin_username,
                node_ip_address=self.master_public_ip_address,
                ssh_port=self.ssh_port
            )
        except CommandExecutionError:
            logger.error_red("No logs have been created at this time")

    @staticmethod
    def _standardize_job_details(start_job_deployment: dict) -> dict:
        # Validate grass_azure_start_job
        optional_key_to_value = {
            "root['tags']": {}
        }
        with open(f"{GlobalPaths.ABS_MARO_GRASS_LIB}/deployments/internal/grass_azure_start_job.yml") as fr:
            start_job_template = yaml.safe_load(fr)
        DeploymentValidator.validate_and_fill_dict(
            template_dict=start_job_template,
            actual_dict=start_job_deployment,
            optional_key_to_value=optional_key_to_value
        )

        # Validate component
        with open(f"{GlobalPaths.ABS_MARO_GRASS_LIB}/deployments/internal/component.yml", "r") as fr:
            start_job_component_template = yaml.safe_load(fr)
        components_details = start_job_deployment["components"]
        for _, component_details in components_details.items():
            DeploymentValidator.validate_and_fill_dict(
                template_dict=start_job_component_template,
                actual_dict=component_details,
                optional_key_to_value={}
            )

        # Init runtime fields
        start_job_deployment["containers"] = {}
        start_job_deployment["id"] = NameCreator.create_job_id()
        for _, component_details in start_job_deployment["components"].items():
            component_details["id"] = NameCreator.create_component_id()

        return start_job_deployment

    # maro grass schedule

    def start_schedule(self, deployment_path: str):
        # Load start_schedule_deployment
        with open(deployment_path, "r") as fr:
            start_schedule_deployment = yaml.safe_load(fr)

        # Standardize start_schedule_deployment
        schedule_details = self._standardize_schedule_details(start_schedule_deployment=start_schedule_deployment)

        # Create schedule
        self.master_api_client.create_schedule(schedule_details=schedule_details)

        logger.info_green(f"Multiple job tickets are sent")

    def stop_schedule(self, schedule_name: str):
        # Stop schedule, TODO: add delete job
        self.master_api_client.stop_schedule(schedule_name=schedule_name)

    @staticmethod
    def _standardize_schedule_details(start_schedule_deployment: dict):
        # Validate grass_azure_start_job
        with open(f"{GlobalPaths.ABS_MARO_GRASS_LIB}/deployments/internal/grass_azure_start_schedule.yml") as fr:
            start_job_template = yaml.safe_load(fr)
        DeploymentValidator.validate_and_fill_dict(
            template_dict=start_job_template,
            actual_dict=start_schedule_deployment,
            optional_key_to_value={}
        )

        # Validate component
        with open(f"{GlobalPaths.ABS_MARO_GRASS_LIB}/deployments/internal/component.yml") as fr:
            start_job_component_template = yaml.safe_load(fr)
        components_details = start_schedule_deployment["components"]
        for _, component_details in components_details.items():
            DeploymentValidator.validate_and_fill_dict(
                template_dict=start_job_component_template,
                actual_dict=component_details,
                optional_key_to_value={}
            )

        return start_schedule_deployment

    # maro grass status

    def status(self, resource_name: str):
        if resource_name == "master":
            return_status = self.master_api_client.get_master()
        elif resource_name == "nodes":
            return_status = self.master_api_client.list_nodes()
        elif resource_name == "containers":
            return_status = self.master_api_client.list_containers()
        else:
            raise BadRequestError(f"Resource '{resource_name}' is unsupported")

        # Print status
        logger.info(
            json.dumps(
                return_status,
                indent=4, sort_keys=True
            )
        )

    # maro grass template

    @staticmethod
    def template(export_path: str):
        # Get templates
        command = f"cp {GlobalPaths.MARO_GRASS_LIB}/deployments/external/* {export_path}"
        _ = SubProcess.run(command)

    # Remote Scripts

    @staticmethod
    def remote_init_build_node_image_vm(admin_username: str, vm_ip_address: str, ssh_port: int):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {ssh_port} {admin_username}@{vm_ip_address} "
            "'python3 ~/init_build_node_image_vm.py'"
        )
        SubProcess.interactive_run(command)

    @staticmethod
    def remote_init_master(admin_username: str, master_ip_address: str, ssh_port: int, cluster_name: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {ssh_port} "
            f"{admin_username}@{master_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.init_master {cluster_name}'"
        )
        SubProcess.interactive_run(command)

    @staticmethod
    def remote_start_master_services(admin_username: str, master_ip_address: str, ssh_port: int, cluster_name: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {ssh_port} "
            f"{admin_username}@{master_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.start_master_agent_service "
            f"{cluster_name}'"
        )
        _ = SubProcess.run(command)
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {ssh_port} "
            f"{admin_username}@{master_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.start_master_api_server_service "
            f"{cluster_name}'"
        )
        _ = SubProcess.run(command)

    def remote_init_node(self, node_name: str, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} {self.admin_username}@{node_ip_address} "
            f"'python3 ~/init_node.py {self.cluster_name} {node_name}'"
        )
        SubProcess.interactive_run(command)

    def remote_start_node_services(self, node_name: str, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} {self.admin_username}@{node_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.node.start_node_agent_service "
            f"{self.cluster_name} {node_name}'"
        )
        _ = SubProcess.run(command)
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} {self.admin_username}@{node_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.node.start_node_api_server_service "
            f"{self.cluster_name}'"
        )
        _ = SubProcess.run(command)

    def remote_stop_node_services(self, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} {self.admin_username}@{node_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.node.stop_node_api_server_service'"
        )
        _ = SubProcess.run(command)
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} {self.admin_username}@{node_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.node.stop_node_agent_service'"
        )
        _ = SubProcess.run(command)

    @staticmethod
    def test_ssh_22_connection(admin_username: str, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no {admin_username}@{node_ip_address} "
            "echo 'Connection established'"
        )
        _ = SubProcess.run(command=command, timeout=5)

    @staticmethod
    def test_ssh_default_port_connection(admin_username: str, node_ip_address: str, ssh_port: int):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {ssh_port} {admin_username}@{node_ip_address} "
            "echo 'Connection established'"
        )
        _ = SubProcess.run(command=command, timeout=5)

    @staticmethod
    def remote_set_ssh_port(admin_username: str, node_ip_address: str, ssh_port: int):
        # Don't have to do the setting if it is assigned 22
        if ssh_port == 22:
            return

        # Set ssh port.
        command = (
            f"ssh -o StrictHostKeyChecking=no {admin_username}@{node_ip_address} "
            f"'echo -e \"Port {ssh_port}\nPort 22\" | sudo tee -a /etc/ssh/sshd_config'"
        )
        _ = SubProcess.run(command)

        # Restart sshd service.
        command = (
            f"ssh -o StrictHostKeyChecking=no {admin_username}@{node_ip_address} "
            "'sudo systemctl restart ssh'"
        )
        _ = SubProcess.run(command)

    @staticmethod
    def retry_connection_and_set_ssh_port(admin_username: str, node_ip_address: str, ssh_port: int) -> bool:
        remain_retries = 20
        while remain_retries > 0:
            try:
                GrassExecutor.test_ssh_default_port_connection(
                    admin_username=admin_username,
                    node_ip_address=node_ip_address,
                    ssh_port=ssh_port
                )
                return True
            except (CliError, TimeoutExpired):
                remain_retries -= 1
                logger.debug(
                    f"Unable to connect to {node_ip_address} with port {ssh_port}, "
                    f"remains {remain_retries} retries"
                )
            try:
                GrassExecutor.test_ssh_22_connection(
                    admin_username=admin_username,
                    node_ip_address=node_ip_address
                )
                GrassExecutor.remote_set_ssh_port(
                    admin_username=admin_username,
                    node_ip_address=node_ip_address,
                    ssh_port=ssh_port
                )
                return True
            except (CliError, TimeoutExpired):
                remain_retries -= 1
                logger.debug(
                    f"Unable to connect to {node_ip_address} with port 22, remains {remain_retries} retries"
                )
            time.sleep(10)
        raise ClusterInternalError(f"Unable to connect to {node_ip_address}")

    # Utils

    @staticmethod
    def _get_md5_checksum(path: str, block_size=128) -> str:
        """ Get md5 checksum of a local file.

        Args:
            path (str): path of the local file.
            block_size (int): size of the reading block, keep it as default value if you are not familiar with it.

        Returns:
            str: md5 checksum str.
        """
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(block_size * md5.block_size), b""):
                md5.update(chunk)
        return md5.hexdigest()
