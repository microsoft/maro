# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import base64
import copy
import json
import os
import time
from multiprocessing.pool import ThreadPool
from subprocess import TimeoutExpired

import yaml

from maro.cli.grass.utils.copy import copy_and_rename, copy_files_from_node, copy_files_to_node
from maro.cli.grass.utils.hash import get_checksum
from maro.cli.grass.utils.params import NodeStatus
from maro.cli.utils.details import load_job_details, load_schedule_details, save_job_details, save_schedule_details
from maro.cli.utils.naming import generate_component_id, generate_job_id, get_valid_file_name
from maro.cli.utils.params import GlobalParams, GlobalPaths
from maro.cli.utils.subprocess import SubProcess
from maro.cli.utils.validation import validate_and_fill_dict
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
        self.master_public_ip_address = self.cluster_details["master"].get("public_ip_address", None)

        # Connection configs
        self.ssh_port = self.cluster_details["connection"]["ssh"]["port"]

    # maro grass node

    def list_node(self):
        # Get nodes details
        nodes_details = self.remote_get_nodes_details()

        # Print details
        logger.info(
            json.dumps(
                nodes_details,
                indent=4, sort_keys=True
            )
        )

    # maro grass image

    def push_image(self, image_name: str, image_path: str, remote_context_path: str, remote_image_name: str):
        # Get images dir
        images_dir = f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/images"

        # Push image
        if image_name:
            new_file_name = get_valid_file_name(image_name)
            abs_image_path = f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/images/{new_file_name}"
            self._save_image(
                image_name=image_name,
                export_path=abs_image_path
            )
            if self._check_checksum_validity(
                local_file_path=abs_image_path,
                remote_file_path=os.path.join(images_dir, image_name)
            ):
                logger.info_green(f"The image file '{new_file_name}' already exists")
                return
            copy_files_to_node(
                local_path=abs_image_path,
                remote_dir=images_dir,
                admin_username=self.admin_username,
                node_ip_address=self.master_public_ip_address,
                ssh_port=self.ssh_port
            )
            self.remote_update_image_files_details()
            self._batch_load_images()
            logger.info_green(f"Image {image_name} is loaded")
        elif image_path:
            file_name = os.path.basename(image_path)
            new_file_name = get_valid_file_name(file_name)
            abs_image_path = f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/images/{new_file_name}"
            copy_and_rename(
                source_path=abs_image_path,
                target_dir=image_path
            )
            if self._check_checksum_validity(
                local_file_path=abs_image_path,
                remote_file_path=os.path.join(images_dir, new_file_name)
            ):
                logger.info_green(f"The image file '{new_file_name}' already exists")
                return
            copy_files_to_node(
                local_path=abs_image_path,
                remote_dir=images_dir,
                admin_username=self.admin_username,
                node_ip_address=self.master_public_ip_address,
                ssh_port=self.ssh_port
            )
            self.remote_update_image_files_details()
            self._batch_load_images()
        elif remote_context_path and remote_image_name:
            self.remote_build_image(
                remote_context_path=remote_context_path,
                remote_image_name=remote_image_name
            )
            self._batch_load_images()
        else:
            raise BadRequestError("Invalid arguments.")

    @staticmethod
    def _save_image(image_name: str, export_path: str):
        # Save image to specific folder
        command = f"docker save '{image_name}' --output '{export_path}'"
        _ = SubProcess.run(command)

    def _batch_load_images(self):
        # Load details
        nodes_details = self.remote_get_nodes_details()

        # build params
        params = []
        for node_name, node_details in nodes_details.items():
            if node_details["state"]["status"] == NodeStatus.RUNNING:
                params.append([
                    node_name,
                    GlobalParams.PARALLELS,
                    node_details["public_ip_address"]
                ])

        # Parallel load image
        with ThreadPool(GlobalParams.PARALLELS) as pool:
            pool.starmap(
                self.remote_load_images,
                params
            )

    def _check_checksum_validity(self, local_file_path: str, remote_file_path: str) -> bool:
        local_checksum = get_checksum(file_path=local_file_path)
        remote_checksum = self.remote_get_checksum(
            file_path=remote_file_path
        )
        return local_checksum == remote_checksum

    # maro grass data

    def push_data(self, local_path: str, remote_path: str):
        if not remote_path.startswith("/"):
            raise FileOperationError(f"Invalid remote path: {remote_path}\nShould be started with '/'.")
        copy_files_to_node(
            local_path=local_path,
            remote_dir=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/data{remote_path}",
            admin_username=self.admin_username,
            node_ip_address=self.master_public_ip_address,
            ssh_port=self.ssh_port
        )

    def pull_data(self, local_path: str, remote_path: str):
        if not remote_path.startswith("/"):
            raise FileOperationError(f"Invalid remote path: {remote_path}\nShould be started with '/'.")
        copy_files_from_node(
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

        # Standardize start_job_deployment
        self._standardize_start_job_deployment(start_job_deployment=start_job_deployment)

        # Start job
        self._start_job(
            job_details=start_job_deployment
        )

    def _start_job(self, job_details: dict):
        logger.info(f"Sending job ticket {job_details['name']}")

        # Load details
        job_name = job_details["name"]

        # Sync mkdir
        self._sync_mkdir(
            path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs/{job_name}",
            node_ip_address=self.master_public_ip_address
        )

        # Save job deployment
        save_job_details(
            cluster_name=self.cluster_name,
            job_name=job_name,
            job_details=job_details
        )

        # Set job id
        self._set_job_id(
            job_name=job_name
        )

        # Sync job details to master
        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs/{job_name}/details.yml",
            remote_dir=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs/{job_name}",
            admin_username=self.admin_username,
            node_ip_address=self.master_public_ip_address,
            ssh_port=self.ssh_port
        )

        # Remote start job
        self.remote_create_job_details(job_name=job_name)
        self.remote_create_pending_job_ticket(job_name=job_name)

        logger.info_green(f"Job ticket {job_details['name']} is sent")

    def stop_job(self, job_name: str):
        # Remote stop job
        self.remote_create_killed_job_ticket(job_name=job_name)
        self.remote_delete_pending_job_ticket(job_name=job_name)

    def list_job(self):
        # Get jobs details
        jobs_details = self.remote_get_jobs_details()

        # Print details
        logger.info(
            json.dumps(
                jobs_details,
                indent=4, sort_keys=True
            )
        )

    def get_job_logs(self, job_name: str, export_dir: str = "./"):
        # Load details
        job_details = load_job_details(
            cluster_name=self.cluster_name,
            job_name=job_name
        )
        job_id = job_details["id"]

        # Copy logs from master
        try:
            copy_files_from_node(
                local_dir=export_dir,
                remote_path=f"~/.maro/logs/{job_id}",
                admin_username=self.admin_username,
                node_ip_address=self.master_public_ip_address,
                ssh_port=self.ssh_port
            )
        except CommandExecutionError:
            logger.error_red("No logs have been created at this time.")

    @staticmethod
    def _standardize_start_job_deployment(start_job_deployment: dict):
        # Validate grass_azure_start_job
        optional_key_to_value = {
            "root['tags']": {}
        }
        with open(f"{GlobalPaths.ABS_MARO_GRASS_LIB}/deployments/internal/grass_azure_start_job.yml") as fr:
            start_job_template = yaml.safe_load(fr)
        validate_and_fill_dict(
            template_dict=start_job_template,
            actual_dict=start_job_deployment,
            optional_key_to_value=optional_key_to_value
        )

        # Validate component
        with open(f"{GlobalPaths.ABS_MARO_GRASS_LIB}/deployments/internal/component.yml", "r") as fr:
            start_job_component_template = yaml.safe_load(fr)
        components_details = start_job_deployment["components"]
        for _, component_details in components_details.items():
            validate_and_fill_dict(
                template_dict=start_job_component_template,
                actual_dict=component_details,
                optional_key_to_value={}
            )

    def _set_job_id(self, job_name: str):
        # Load details
        job_details = load_job_details(cluster_name=self.cluster_name, job_name=job_name)

        # Set cluster id
        job_details["id"] = generate_job_id()

        # Set component id
        for component, component_details in job_details["components"].items():
            component_details["id"] = generate_component_id()

        # Save details
        save_job_details(
            cluster_name=self.cluster_name,
            job_name=job_name,
            job_details=job_details
        )

    # maro grass schedule

    def start_schedule(self, deployment_path: str):
        # Load start_schedule_deployment
        with open(deployment_path, "r") as fr:
            start_schedule_deployment = yaml.safe_load(fr)

        # Standardize start_schedule_deployment
        self._standardize_start_schedule_deployment(start_schedule_deployment=start_schedule_deployment)
        schedule_name = start_schedule_deployment["name"]

        # Sync mkdir
        self._sync_mkdir(
            path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/schedules/{schedule_name}",
            node_ip_address=self.master_public_ip_address
        )

        # Save schedule deployment
        save_schedule_details(
            cluster_name=self.cluster_name,
            schedule_name=schedule_name,
            schedule_details=start_schedule_deployment
        )

        # Start jobs
        for job_name in start_schedule_deployment["job_names"]:
            job_details = self._build_job_details(
                schedule_details=start_schedule_deployment,
                job_name=job_name
            )

            self._start_job(
                job_details=job_details
            )

    def stop_schedule(self, schedule_name: str):
        # Load details
        schedule_details = load_schedule_details(cluster_name=self.cluster_name, schedule_name=schedule_name)
        job_names = schedule_details["job_names"]

        for job_name in job_names:
            # Load job details
            job_details = load_job_details(cluster_name=self.cluster_name, job_name=job_name)
            job_schedule_tag = job_details["tags"]["schedule"]

            # Remote stop job
            if job_schedule_tag == schedule_name:
                self.remote_create_killed_job_ticket(job_name=job_name)
                self.remote_delete_pending_job_ticket(job_name=job_name)

    @staticmethod
    def _standardize_start_schedule_deployment(start_schedule_deployment: dict):
        # Validate grass_azure_start_job
        with open(f"{GlobalPaths.ABS_MARO_GRASS_LIB}/deployments/internal/grass_azure_start_schedule.yml") as fr:
            start_job_template = yaml.safe_load(fr)
        validate_and_fill_dict(
            template_dict=start_job_template,
            actual_dict=start_schedule_deployment,
            optional_key_to_value={}
        )

        # Validate component
        with open(f"{GlobalPaths.ABS_MARO_GRASS_LIB}/deployments/internal/component.yml") as fr:
            start_job_component_template = yaml.safe_load(fr)
        components_details = start_schedule_deployment["components"]
        for _, component_details in components_details.items():
            validate_and_fill_dict(
                template_dict=start_job_component_template,
                actual_dict=component_details,
                optional_key_to_value={}
            )

    @staticmethod
    def _build_job_details(schedule_details: dict, job_name: str) -> dict:
        schedule_name = schedule_details["name"]

        job_details = copy.deepcopy(schedule_details)
        job_details["name"] = job_name
        job_details["tags"] = {
            "schedule": schedule_name
        }
        job_details.pop("job_names")

        return job_details

    # maro grass status

    def status(self, resource_name: str):
        if resource_name == "master":
            return_status = self.remote_get_master_details()
        elif resource_name == "nodes":
            return_status = self.remote_get_nodes_details()
        elif resource_name == "containers":
            return_status = self.remote_get_containers_details()
        else:
            raise BadRequestError(f"Resource '{resource_name}' is unsupported.")

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

    # Remote utils

    def remote_build_image(self, remote_context_path: str, remote_image_name: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.build_image "
            f"{self.cluster_name} {remote_context_path} {remote_image_name}'"
        )
        _ = SubProcess.run(command)

    def remote_clean(self, parallels: int):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.clean {self.cluster_name} {parallels}'"
        )
        _ = SubProcess.run(command)

    def remote_get_checksum(self, file_path: str) -> str:
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.get_checksum {file_path}'"
        )
        return_str = SubProcess.run(command)
        return return_str

    def remote_get_jobs_details(self):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.get_jobs_details {self.cluster_name}'"
        )
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_master_details(self):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.get_master_details {self.cluster_name}'"
        )
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_node_details(self, node_name: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.get_node_details {self.cluster_name} {node_name}'"
        )
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_nodes_details(self):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.get_nodes_details {self.cluster_name}'"
        )
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_containers_details(self):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.get_containers_details {self.cluster_name}'"
        )
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    def remote_get_public_key(self, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} {self.admin_username}@{node_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.get_public_key'"
        )
        return_str = SubProcess.run(command).strip("\n")
        logger.debug(return_str)
        return return_str

    def remote_init_build_node_image_vm(self, vm_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} {self.admin_username}@{vm_ip_address} "
            "'python3 ~/init_build_node_image_vm.py'"
        )
        SubProcess.interactive_run(command)

    def remote_init_master(self):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.init_master {self.cluster_name}'"
        )
        SubProcess.interactive_run(command)

    def remote_init_node(self, node_name: str, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} {self.admin_username}@{node_ip_address} "
            f"'python3 ~/init_node.py {self.cluster_name} {node_name}'"
        )
        SubProcess.interactive_run(command)

    def remote_mkdir(self, node_ip_address: str, path: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} {self.admin_username}@{node_ip_address} "
            f"'mkdir -p {path}'"
        )
        SubProcess.run(command)

    def remote_load_images(self, node_name: str, parallels: int, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} {self.admin_username}@{node_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.load_images "
            f"{self.cluster_name} {node_name} {parallels}'"
        )
        SubProcess.interactive_run(command)

    def remote_load_master_agent_service(self):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.load_master_agent_service {self.cluster_name}'"
        )
        _ = SubProcess.run(command)

    def remote_load_node_agent_service(self, node_name: str, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} {self.admin_username}@{node_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.load_node_agent_service "
            f"{self.cluster_name} {node_name}'"
        )
        _ = SubProcess.run(command)

    def remote_create_pending_job_ticket(self, job_name: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.create_pending_job_ticket "
            f"{self.cluster_name} {job_name}'"
        )
        _ = SubProcess.run(command)

    def remote_create_job_details(self, job_name: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.create_job_details "
            f"{self.cluster_name} {job_name}'"
        )
        _ = SubProcess.run(command)

    def remote_create_killed_job_ticket(self, job_name: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.create_killed_job_ticket "
            f"{self.cluster_name} {job_name}'"
        )
        _ = SubProcess.run(command)

    def remote_delete_pending_job_ticket(self, job_name: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.delete_pending_job_ticket "
            f"{self.cluster_name} {job_name}'"
        )
        _ = SubProcess.run(command)

    def remote_set_master_details(self, master_details: dict):
        master_details_b64 = base64.b64encode(json.dumps(master_details).encode("utf8")).decode('utf8')
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.set_master_details "
            f"{self.cluster_name} {master_details_b64}'"
        )
        _ = SubProcess.run(command)

    def remote_set_node_details(self, node_name: str, node_details: dict):
        node_details_b64 = base64.b64encode(json.dumps(node_details).encode("utf8")).decode('utf8')
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.set_node_details "
            f"{self.cluster_name} {node_name} {node_details_b64}'"
        )
        _ = SubProcess.run(command)

    def remote_update_image_files_details(self):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.update_image_files_details "
            f"{self.cluster_name}'"
        )
        _ = SubProcess.run(command)

    def remote_update_node_status(self, node_name: str, action: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.update_node_status "
            f"{self.cluster_name} {node_name} {action}'"
        )
        _ = SubProcess.run(command)

    def test_ssh_22_connection(self, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} "
            "echo 'Connection established'"
        )
        _ = SubProcess.run(command=command, timeout=5)

    def test_ssh_default_port_connection(self, node_ip_address: str):
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {self.ssh_port} {self.admin_username}@{node_ip_address} "
            "echo 'Connection established'"
        )
        _ = SubProcess.run(command=command, timeout=5)

    def remote_set_ssh_port(self, node_ip_address: str):
        # Don't have to do the setting if it is assigned 22
        if self.ssh_port == 22:
            return

        # Set ssh port.
        command = (
            f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} "
            f"'echo -e \"Port {self.ssh_port}\nPort 22\" | sudo tee -a /etc/ssh/sshd_config'"
        )
        _ = SubProcess.run(command)

        # Restart sshd service.
        command = (
            f"ssh -o StrictHostKeyChecking=no {self.admin_username}@{node_ip_address} "
            "'sudo systemctl restart ssh'"
        )
        _ = SubProcess.run(command)

    def retry_connection_and_set_ssh_port(self, node_ip_address: str) -> bool:
        remain_retries = 20
        while remain_retries > 0:
            try:
                self.test_ssh_default_port_connection(node_ip_address=node_ip_address)
                return True
            except (CliError, TimeoutExpired):
                remain_retries -= 1
                logger.debug(
                    f"Unable to connect to {node_ip_address} with port {self.ssh_port}, "
                    f"remains {remain_retries} retries."
                )
            try:
                self.test_ssh_22_connection(node_ip_address=node_ip_address)
                self.remote_set_ssh_port(node_ip_address=node_ip_address)
                return True
            except (CliError, TimeoutExpired):
                remain_retries -= 1
                logger.debug(
                    f"Unable to connect to {node_ip_address} with port 22, remains {remain_retries} retries."
                )
            time.sleep(10)
        raise ClusterInternalError(f"Unable to connect to {node_ip_address}.")

    def remote_delete_master_details(self):
        command = (
            "ssh -o StrictHostKeyChecking=no "
            f"{self.admin_username}@{self.master_public_ip_address} "
            f"'cd {GlobalPaths.MARO_GRASS_LIB}; python3 -m scripts.master.delete_master_details "
            f"{self.cluster_name} '"
        )
        _ = SubProcess.run(command)

    # Utils

    def _sync_mkdir(self, path: str, node_ip_address: str):
        """Mkdir synchronously at local and remote.

        Args:
            path (str): path of the file, should be a string with an initial component of ~ or ~user
            node_ip_address (str): ip address of the remote node
        """
        # Create local dir
        os.makedirs(os.path.expanduser(path), exist_ok=True)

        # Create remote dir
        self.remote_mkdir(node_ip_address=node_ip_address, path=path)
