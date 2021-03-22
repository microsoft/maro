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
from maro.cli.grass.utils.params import GrassPaths, JobStatus, UserRole
from maro.cli.utils.deployment_validator import DeploymentValidator
from maro.cli.utils.details_reader import DetailsReader
from maro.cli.utils.details_writer import DetailsWriter
from maro.cli.utils.name_creator import NameCreator
from maro.cli.utils.params import GlobalPaths
from maro.cli.utils.subprocess import Subprocess
from maro.utils.exception.cli_exception import (
    BadRequestError, CliError, ClusterInternalError, CommandExecutionError, FileOperationError
)
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class GrassExecutor:
    """Shared methods of Grass Mode.

    Including image, job, schedule, and other shared operations.
    """

    def __init__(self, cluster_details: dict):
        self.cluster_details = cluster_details

        # Upper configs
        self.cluster_name = self.cluster_details["name"]
        self.cluster_id = self.cluster_details["id"]

        # User configs
        self.user_details = DetailsReader.load_default_user_details(cluster_name=self.cluster_name)

        # Master configs
        self.master_username = self.cluster_details["master"]["username"]
        self.master_public_ip_address = self.cluster_details["master"]["public_ip_address"]
        self.master_private_ip_address = self.cluster_details["master"]["private_ip_address"]
        self.master_redis_port = self.cluster_details["master"]["redis"]["port"]
        self.master_hostname = self.cluster_details["master"]["hostname"]
        self.master_api_client = MasterApiClientV1(
            master_hostname=self.master_public_ip_address,
            master_api_server_port=self.cluster_details["master"]["api_server"]["port"],
            user_id=self.user_details["id"],
            master_to_dev_encryption_private_key=self.user_details["master_to_dev_encryption_private_key"],
            dev_to_master_encryption_public_key=self.user_details["dev_to_master_encryption_public_key"],
            dev_to_master_signing_private_key=self.user_details["dev_to_master_signing_private_key"]
        )
        self.master_ssh_port = self.cluster_details["master"]["ssh"]["port"]
        self.master_api_server_port = self.cluster_details["master"]["api_server"]["port"]

    # maro grass create

    @staticmethod
    def _init_master(cluster_details: dict) -> None:
        """Init MARO Master VM.

        Args:
            cluster_details (dict): details of the MARO Cluster.

        Returns:
            None.
        """
        logger.info("Initializing Master VM")

        # Make sure master is able to connect
        GrassExecutor.retry_connection(
            node_username=cluster_details["master"]["username"],
            node_hostname=cluster_details["master"]["public_ip_address"],
            node_ssh_port=cluster_details["master"]["ssh"]["port"]
        )

        DetailsWriter.save_cluster_details(
            cluster_name=cluster_details["name"],
            cluster_details=cluster_details
        )

        # Copy required files
        local_path_to_remote_dir = {
            GrassPaths.ABS_MARO_GRASS_LIB: f"{GlobalPaths.MARO_SHARED}/lib",
            f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_details['name']}": f"{GlobalPaths.MARO_SHARED}/clusters"
        }
        for local_path, remote_dir in local_path_to_remote_dir.items():
            FileSynchronizer.copy_files_to_node(
                local_path=local_path,
                remote_dir=remote_dir,
                node_username=cluster_details["master"]["username"],
                node_hostname=cluster_details["master"]["public_ip_address"],
                node_ssh_port=cluster_details["master"]["ssh"]["port"]
            )

        # Remote init master
        GrassExecutor.remote_init_master(
            master_username=cluster_details["master"]["username"],
            master_hostname=cluster_details["master"]["public_ip_address"],
            master_ssh_port=cluster_details["master"]["ssh"]["port"],
            cluster_name=cluster_details["name"]
        )
        # Gracefully wait
        time.sleep(10)

        logger.info_green("Master VM is initialized")

    @staticmethod
    def _create_user(cluster_details: dict) -> None:
        """Create the admin MARO User for the MARO Cluster, and assign the user as the default user of current machine.

        Args:
            cluster_details (dict): details of the MARO Cluster.

        Returns:
            None.
        """
        # Remote create user
        user_details = GrassExecutor.remote_create_user(
            master_username=cluster_details["master"]["username"],
            master_hostname=cluster_details["master"]["public_ip_address"],
            master_ssh_port=cluster_details["master"]["ssh"]["port"],
            user_id=cluster_details["user"]["admin_id"],
            user_role=UserRole.ADMIN
        )

        # Update user_details, "admin_id" change to "id"
        cluster_details["user"] = user_details

        # Save dev_to_master private key
        os.makedirs(
            name=f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_details['name']}/users/{user_details['id']}",
            exist_ok=True
        )
        with open(
            file=f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_details['name']}/users/{user_details['id']}/user_details",
            mode="w"
        ) as fw:
            yaml.safe_dump(
                data=user_details,
                stream=fw
            )

        # Save default user
        with open(file=f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_details['name']}/users/default_user", mode="w") as fw:
            fw.write(user_details["id"])

    # maro grass node

    def list_node(self) -> None:
        """Print node details to command line.

        Returns:
            None.
        """
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

    def push_image(self, image_name: str, image_path: str, remote_context_path: str, remote_image_name: str) -> None:
        """Push docker image from local to the MARO Cluster.

        Args:
            image_name (str): name of the image.
            image_path (str): path of the image file.
            remote_context_path (str): path of the remote context (for remote build).
            remote_image_name (str): name of the image (for remote build).

        Returns:
            None.
        """
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
            # Use md5_checksum to skip existed image file.
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
                remote_dir=f"{GlobalPaths.MARO_SHARED}/clusters/{self.cluster_name}/image_files",
                node_username=self.master_username,
                node_hostname=self.master_public_ip_address,
                node_ssh_port=self.master_ssh_port
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

    def push_data(self, local_path: str, remote_path: str) -> None:
        """Push data from local to remote MARO Cluster.

        Args:
            local_path (str): path of the local file.
            remote_path (str): path of the remote folder.

        Returns:
            None.
        """
        if not remote_path.startswith("/"):
            raise FileOperationError(f"Invalid remote path: {remote_path}\nShould be started with '/'")
        FileSynchronizer.copy_files_to_node(
            local_path=local_path,
            remote_dir=f"{GlobalPaths.MARO_SHARED}/clusters/{self.cluster_name}/data{remote_path}",
            node_username=self.master_username,
            node_hostname=self.master_public_ip_address,
            node_ssh_port=self.master_ssh_port
        )

    def pull_data(self, local_path: str, remote_path: str) -> None:
        """Pull data from remote MARO Cluster to local.

        Args:
            local_path (str): path of the local folder.
            remote_path (str): path of the remote file.

        Returns:
            None.
        """
        if not remote_path.startswith("/"):
            raise FileOperationError(f"Invalid remote path: {remote_path}\nShould be started with '/'")
        FileSynchronizer.copy_files_from_node(
            local_dir=local_path,
            remote_path=f"{GlobalPaths.MARO_SHARED}/clusters/{self.cluster_name}/data{remote_path}",
            node_username=self.master_username,
            node_hostname=self.master_public_ip_address,
            node_ssh_port=self.master_ssh_port
        )

    # maro grass job

    def start_job(self, deployment_path: str) -> None:
        """Start a MARO Job with start_job_deployment.

        Args:
            deployment_path (str): path of the start_job_deployment.

        Returns:
            None.
        """
        # Load start_job_deployment
        with open(deployment_path, "r") as fr:
            start_job_deployment = yaml.safe_load(fr)

        self._start_job(start_job_deployment=start_job_deployment)

    def _start_job(self, start_job_deployment: dict) -> None:
        """Start a MARO Job by sending job_details to the MARO Cluster.

        Args:
            start_job_deployment (dict): raw start_job_deployment.

        Returns:
            None.
        """
        # Standardize start_job_deployment
        job_details = self._standardize_job_details(start_job_deployment=start_job_deployment)

        # Create job
        logger.info(f"Sending job ticket '{start_job_deployment['name']}'")
        self.master_api_client.create_job(job_details=job_details)
        logger.info_green(f"Job ticket '{job_details['name']}' is sent")

    def stop_job(self, job_name: str) -> None:
        """Stop a MARO Job.

        Args:
            job_name (str): name of the MARO Job.

        Returns:
            None.
        """
        # Delete job
        self.master_api_client.delete_job(job_name=job_name)

    def list_job(self) -> None:
        """Print job details to command line.

        Returns:
            None.
        """
        # Get jobs details
        jobs_details = self.master_api_client.list_jobs()

        # Print details
        logger.info(
            json.dumps(
                jobs_details,
                indent=4, sort_keys=True
            )
        )

    def get_job_logs(self, job_name: str, export_dir: str = "./") -> None:
        """Pull logs of a job from remote MARO Cluster to local.

        Args:
            job_name (str): name of the job.
            export_dir (path): path of the local exported folder.

        Returns:
            None.
        """
        # Load details
        job_details = self.master_api_client.get_job(job_name=job_name)

        # Copy logs from master
        try:
            FileSynchronizer.copy_files_from_node(
                local_dir=export_dir,
                remote_path=f"{GlobalPaths.MARO_SHARED}/clusters/{self.cluster_name}/logs/{job_details['id']}",
                node_username=self.master_username,
                node_hostname=self.master_public_ip_address,
                node_ssh_port=self.master_ssh_port
            )
        except CommandExecutionError:
            logger.error_red("No logs have been created at this time")

    @staticmethod
    def _standardize_job_details(start_job_deployment: dict) -> dict:
        """Standardize job_details.

        Args:
            start_job_deployment (dict): start_job_deployment of grass/azure.
                See lib/deployments/internal for reference.

        Returns:
            dict: standardized job_details.
        """
        # Validate grass_azure_start_job
        optional_key_to_value = {
            "root['tags']": {}
        }
        with open(f"{GrassPaths.ABS_MARO_GRASS_LIB}/deployments/internal/grass_azure_start_job.yml") as fr:
            start_job_template = yaml.safe_load(fr)
        DeploymentValidator.validate_and_fill_dict(
            template_dict=start_job_template,
            actual_dict=start_job_deployment,
            optional_key_to_value=optional_key_to_value
        )

        # Validate component
        with open(f"{GrassPaths.ABS_MARO_GRASS_LIB}/deployments/internal/component.yml", "r") as fr:
            start_job_component_template = yaml.safe_load(fr)
        components_details = start_job_deployment["components"]
        for _, component_details in components_details.items():
            DeploymentValidator.validate_and_fill_dict(
                template_dict=start_job_component_template,
                actual_dict=component_details,
                optional_key_to_value={}
            )

        # Init runtime fields
        start_job_deployment["status"] = JobStatus.PENDING
        start_job_deployment["containers"] = {}
        start_job_deployment["id"] = NameCreator.create_job_id()
        for _, component_details in start_job_deployment["components"].items():
            component_details["id"] = NameCreator.create_component_id()

        return start_job_deployment

    # maro grass schedule

    def start_schedule(self, deployment_path: str) -> None:
        """Start a MARO Schedule with start_schedule_deployment.

        Args:
            deployment_path (str): path of the start_schedule_deployment.

        Returns:
            None.
        """
        # Load start_schedule_deployment
        with open(deployment_path, "r") as fr:
            start_schedule_deployment = yaml.safe_load(fr)

        # Standardize start_schedule_deployment
        schedule_details = self._standardize_schedule_details(start_schedule_deployment=start_schedule_deployment)

        # Create schedule
        self.master_api_client.create_schedule(schedule_details=schedule_details)

        logger.info_green("Multiple job tickets are sent")

    def stop_schedule(self, schedule_name: str) -> None:
        """Stop a MARO Schedule.

        Args:
            schedule_name (str): name of the schedule.

        Returns:
            None.
        """
        # Stop schedule, TODO: add delete job
        self.master_api_client.stop_schedule(schedule_name=schedule_name)

    @staticmethod
    def _standardize_schedule_details(start_schedule_deployment: dict) -> dict:
        """Standardize schedule_details.

        Args:
            start_schedule_deployment (dict): start_schedule_deployment of grass/azure.
                See lib/deployments/internal for reference.

        Returns:
            dict: standardized schedule_details.
        """
        # Validate grass_azure_start_job
        with open(f"{GrassPaths.ABS_MARO_GRASS_LIB}/deployments/internal/grass_azure_start_schedule.yml") as fr:
            start_job_template = yaml.safe_load(fr)
        DeploymentValidator.validate_and_fill_dict(
            template_dict=start_job_template,
            actual_dict=start_schedule_deployment,
            optional_key_to_value={}
        )

        # Validate component
        with open(f"{GrassPaths.ABS_MARO_GRASS_LIB}/deployments/internal/component.yml") as fr:
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

    def status(self, resource_name: str) -> None:
        """Print details of specific resources (master/nodes/containers).

        Args:
            resource_name (str): name of the resource.

        Returns:
            None.
        """
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
    def template(export_path: str) -> None:
        """Export deployment template of grass mode.

        Args:
            export_path (str): location to export the templates.

        Returns:
            None.
        """
        # Get templates
        command = f"cp {GrassPaths.MARO_GRASS_LIB}/deployments/external/* {export_path}"
        _ = Subprocess.run(command=command)

    # Remote Scripts

    @staticmethod
    def remote_init_build_node_image_vm(node_username: str, node_hostname: str, node_ssh_port: int) -> None:
        """Remote init Build Node Image VM.

        Exec /lib/scripts/build_node_image_vm/init_build_node_image_vm.py remotely.

        Args:
            node_username (str): username of the vm.
            node_hostname (str): hostname of the vm.
            node_ssh_port (int): ssh port of the vm.

        Returns:
            None.
        """
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {node_ssh_port} {node_username}@{node_hostname} "
            "'python3 ~/init_build_node_image_vm.py'"
        )
        Subprocess.interactive_run(command=command)

    @staticmethod
    def remote_init_master(master_username: str, master_hostname: str, master_ssh_port: int, cluster_name: str) -> None:
        """Remote init MARO Master.

        Exec /lib/scripts/master/init_master.py remotely.

        Args:
            master_username (str): username of the MARO Master VM.
            master_hostname (str): hostname of the MARO Master VM.
            master_ssh_port (int): ssh port of the MARO Master VM.
            cluster_name (str): name of the MARO Cluster.

        Returns:
            None.
        """
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {master_ssh_port} {master_username}@{master_hostname} "
            f"'cd {GlobalPaths.MARO_SHARED}/lib/grass; python3 -m scripts.master.init_master {cluster_name}'"
        )
        Subprocess.interactive_run(command=command)

    @staticmethod
    def remote_delete_master(master_username: str, master_hostname: str, master_ssh_port: int) -> None:
        """Remote delete MARO Master.

        Exec /lib/scripts/master/delete_master.py remotely.

        Args:
            master_username (str): username of the MARO Master VM.
            master_hostname (str): hostname of the MARO Master VM.
            master_ssh_port (int): ssh port of the MARO Master VM.

        Returns:
            None.
        """
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {master_ssh_port} {master_username}@{master_hostname} "
            f"'python3 {GlobalPaths.MARO_LOCAL}/scripts/delete_master.py'"
        )
        Subprocess.interactive_run(command=command)

    @staticmethod
    def remote_create_user(
        master_username: str, master_hostname: str, master_ssh_port: int,
        user_id: str, user_role: str
    ) -> dict:
        """Remote create MARO User.

        Exec /lib/scripts/master/create_user.py remotely.

        Args:
            master_username (str): username of the MARO Master VM.
            master_hostname (str): hostname of the MARO Master VM.
            master_ssh_port (int): ssh port of the MARO Master VM.
            user_id (str): id of the MARO User.
            user_role (str): role of the MARO User, currently we only have 'admin' at this time.

        Returns:
            dict: details of the created MARO User.
        """
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {master_ssh_port} {master_username}@{master_hostname} "
            f"'cd {GlobalPaths.MARO_SHARED}/lib/grass; python3 -m scripts.master.create_user "
            f"{user_id} {user_role}'"
        )
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    @staticmethod
    def remote_join_cluster(
        node_username: str, node_hostname: str, node_ssh_port: int,
        master_private_ip_address: str, master_api_server_port: int, deployment_path: str
    ) -> None:
        """Remote join cluster.

        Install required runtime env first,
        then download the /lib/scripts/node/join_cluster.py from master_api_server, and execute remotely.

        Args:
            node_username (str): username of the MARO Node VM.
            node_hostname (str): hostname of the MARO Node VM.
            node_ssh_port (str): ssh port of the MARO Node VM.
            master_private_ip_address (str): private ip address of the MARO Master VM,
                (master and nodes must in the same virtual network).
            master_api_server_port (int): port of the master_api_server.
            deployment_path (str): path of the join_cluster_deployment.

        Returns:
            None.
        """
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {node_ssh_port} {node_username}@{node_hostname} "
            "'export DEBIAN_FRONTEND=noninteractive; "
            "sudo -E apt update; "
            "sudo -E apt install -y python3-pip; "
            "pip3 install deepdiff redis pyyaml; "
            f"curl -s GET http://{master_private_ip_address}:{master_api_server_port}/v1/joinClusterScript | "
            f"python3 - {deployment_path}'"
        )
        Subprocess.interactive_run(command=command)

    @staticmethod
    def local_join_cluster(master_hostname: str, master_private_ip_address: int, deployment_path: str) -> None:
        """Local join cluster.

        Download the /lib/scripts/node/join_cluster.py from master_api_server, and execute it locally.

        Args:
            master_hostname (str): hostname of the MARO Master VM.
            master_private_ip_address (str): private ip address of the MARO Master VM,
                (master and nodes must in the same virtual network).
            deployment_path (str): path of the join_cluster_deployment.

        Returns:
            None.
        """
        command = (
            f"'curl -s GET http://{master_hostname}:{master_private_ip_address}/v1/joinClusterScript | "
            f"python3 - {deployment_path}'"
        )
        Subprocess.interactive_run(command=command)

    @staticmethod
    def remote_leave_cluster(node_username: str, node_hostname: str, node_ssh_port: int) -> None:
        """Remote leave cluster.

        Exec /lib/scripts/node/activate_leave_cluster.py

        Args:
            node_username (str): username of the MARO Node VM.
            node_hostname (str): hostname of the MARO Node VM.
            node_ssh_port (str): ssh port of the MARO Node VM.

        Returns:
            None.
        """
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {node_ssh_port} {node_username}@{node_hostname} "
            f"'python3 ~/.maro-local/scripts/activate_leave_cluster.py'"
        )
        Subprocess.interactive_run(command=command)

    @staticmethod
    def local_leave_cluster() -> None:
        """Local leave cluster.

        Exec /lib/scripts/node/activate_leave_cluster.py

        Returns:
            None.
        """
        command = "python3 ~/.maro-local/scripts/activate_leave_cluster.py"
        Subprocess.interactive_run(command=command)

    @staticmethod
    def test_ssh_default_port_connection(node_username: str, node_hostname: str, node_ssh_port: int) -> None:
        """Test ssh connection.

        Args:
            node_username (str): username of the MARO Node VM.
            node_hostname (str): hostname of the MARO Node VM.
            node_ssh_port (str): ssh port of the MARO Node VM.

        Raises:
            CliError / TimeoutExpired: if the connection is failed.

        Returns:
            None.
        """
        command = (
            f"ssh -o StrictHostKeyChecking=no -p {node_ssh_port} {node_username}@{node_hostname} "
            "echo 'Connection established'"
        )
        _ = Subprocess.run(command=command, timeout=5)

    @staticmethod
    def retry_connection(node_username: str, node_hostname: str, node_ssh_port: int) -> None:
        """Retry SSH connection until it is connectable.

        Args:
            node_username (str): username of the MARO Node VM.
            node_hostname (str): hostname of the MARO Node VM.
            node_ssh_port (str): ssh port of the MARO Node VM.

        Raises:
            ClusterInternalError: if the connection is failed.

        Returns:
            None.
        """
        remain_retries = 20
        while remain_retries > 0:
            try:
                GrassExecutor.test_ssh_default_port_connection(
                    node_ssh_port=node_ssh_port,
                    node_username=node_username,
                    node_hostname=node_hostname
                )
                return
            except (CliError, TimeoutExpired):
                remain_retries -= 1
                logger.debug(
                    f"Unable to connect to {node_hostname} with port {node_ssh_port}, "
                    f"remains {remain_retries} retries"
                )
                time.sleep(5)
        raise ClusterInternalError(f"Unable to connect to {node_hostname} with port {node_ssh_port}")

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

    # Visible

    def get_job_details(self):
        return self.master_api_client.list_jobs()

    def get_job_queue(self):
        return self.master_api_client.get_job_queue()

    def get_resource(self):
        return self.master_api_client.get_static_resource_info()

    def get_resource_usage(self, previous_length: int):
        return self.master_api_client.get_dynamic_resource_info(previous_length)
