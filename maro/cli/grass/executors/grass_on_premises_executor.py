# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import secrets
import string
from shutil import rmtree

import yaml

from maro.cli.grass.executors.grass_executor import GrassExecutor
from maro.cli.grass.utils.file_synchronizer import FileSynchronizer
from maro.cli.utils.deployment_validator import DeploymentValidator
from maro.cli.utils.details_reader import DetailsReader
from maro.cli.utils.details_writer import DetailsWriter
from maro.cli.utils.name_creator import NameCreator
from maro.cli.utils.params import GlobalParams, GlobalPaths
from maro.cli.utils.subprocess import SubProcess
from maro.utils.exception.cli_exception import CliError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class GrassOnPremisesExecutor(GrassExecutor):

    def __init__(self, cluster_name: str):
        super().__init__(cluster_details=DetailsReader.load_cluster_details(cluster_name=cluster_name))

    @staticmethod
    def build_cluster_details(create_deployment: dict):
        # Standardize create deployment
        GrassOnPremisesExecutor._standardize_create_deployment(create_deployment=create_deployment)

        # Create user account
        logger.info("Now is going to create an user account for maro cluster node.")
        if "super_user" in create_deployment["user"]:
            super_user = create_deployment["user"]["super_user"]
        else:
            super_user = ""
        GrassOnPremisesExecutor.create_user(
            admin_username=super_user,
            maro_user=create_deployment["user"]["admin_username"],
            ip_address=create_deployment["master"]["public_ip_address"],
            pubkey=create_deployment["user"]["admin_public_key"],
            ssh_port=create_deployment["connection"]["ssh"]["port"]
        )

        # Get cluster name and save details
        cluster_name = create_deployment["name"]
        if os.path.isdir(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}")):
            raise CliError(f"Cluster {cluster_name} already exist.")
        os.makedirs(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}"))
        DetailsWriter.save_cluster_details(
            cluster_name=cluster_name,
            cluster_details=create_deployment
        )

    @staticmethod
    def _standardize_create_deployment(create_deployment: dict):
        samba_password = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20))
        optional_key_to_value = {
            "root['master']['redis']": {"port": GlobalParams.DEFAULT_REDIS_PORT},
            "root['master']['redis']['port']": GlobalParams.DEFAULT_REDIS_PORT,
            "root['master']['fluentd']": {"port": GlobalParams.DEFAULT_FLUENTD_PORT},
            "root['master']['fluentd']['port']": GlobalParams.DEFAULT_FLUENTD_PORT,
            "root['master']['samba']": {"password": samba_password},
            "root['master']['samba']['password']": samba_password,
            "root['connection']": {"ssh": {"port": GlobalParams.DEFAULT_SSH_PORT}},
            "root['connection']['ssh']": {"port": GlobalParams.DEFAULT_SSH_PORT},
            "root['connection']['ssh']['port']": GlobalParams.DEFAULT_SSH_PORT
        }
        with open(
            os.path.expanduser(
                f"{GlobalPaths.MARO_GRASS_LIB}/deployments/internal/grass-on-premises-create.yml")
        ) as fr:
            create_deployment_template = yaml.safe_load(fr)
        DeploymentValidator.validate_and_fill_dict(
            template_dict=create_deployment_template,
            actual_dict=create_deployment,
            optional_key_to_value=optional_key_to_value
        )

    def create(self):
        logger.info("Creating cluster")

        # Start creating
        try:
            self._set_cluster_id()
            self._set_master_info()
            self._init_master()
        except Exception as e:
            # If failed, remove details folder, then raise
            rmtree(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}"))
            raise CliError(f"Failure to create cluster, due to {e}")

        logger.info_green(f"Cluster {self.cluster_name} has been created.")

    def _set_cluster_id(self):
        # Load details
        cluster_details = self.cluster_details

        # Set cluster id
        cluster_details["id"] = NameCreator.create_cluster_id()

        # Save details
        DetailsWriter.save_cluster_details(
            cluster_name=self.cluster_name,
            cluster_details=cluster_details
        )

    def _create_path_in_list(self, target_ip: str, path_list):
        for path_to_create in path_list:
            self.remote_mkdir(
                path=path_to_create,
                node_ip_address=target_ip
            )

    def _set_master_info(self):
        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details["id"]
        master_details = cluster_details["master"]
        hostname = cluster_details["master"]["public_ip_address"]
        master_details["private_ip_address"] = cluster_details["master"]["public_ip_address"]
        master_details["hostname"] = hostname
        master_details["resource_name"] = f"{cluster_id}-master-vm"
        admin_username = cluster_details["user"]["admin_username"]
        public_ip_address = cluster_details["master"]["public_ip_address"]
        logger.info_green(f"You can login to your master node with: ssh {admin_username}@{public_ip_address}")

    def _init_master(self):
        logger.info("Initializing master node")

        # Load details
        cluster_details = self.cluster_details
        master_details = cluster_details["master"]
        admin_username = cluster_details["user"]["admin_username"]
        master_public_ip_address = cluster_details["master"]["public_ip_address"]
        ssh_port = cluster_details["connection"]["ssh"]["port"]

        # Make sure master is able to connect
        self.retry_connection_and_set_ssh_port(node_ip_address=master_public_ip_address)

        # Create folders
        path_list = {
            GlobalPaths.MARO_GRASS_LIB,
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}",
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/data",
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/image_files",
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs",
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/schedules"
        }
        self._create_path_in_list(master_public_ip_address, path_list)

        # Copy required files
        FileSynchronizer.copy_files_to_node(
            local_path=GlobalPaths.MARO_GRASS_LIB,
            remote_dir=GlobalPaths.MARO_LIB,
            admin_username=admin_username, node_ip_address=master_public_ip_address, ssh_port=ssh_port
        )
        FileSynchronizer.copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}",
            remote_dir=GlobalPaths.MARO_CLUSTERS,
            admin_username=admin_username, node_ip_address=master_public_ip_address, ssh_port=ssh_port
        )

        # Remote init master
        self.remote_init_master()

        # Load master agent service
        self.remote_start_master_services()

        # Save details
        DetailsWriter.save_cluster_details(
            cluster_name=self.cluster_name,
            cluster_details=cluster_details
        )
        self.remote_create_master_details(master_details=cluster_details["master"])

        logger.info_green("Master node is initialized")

    def delete(self):
        # Load details
        cluster_name = self.cluster_name
        logger.info(f"Deleting cluster {cluster_name}")

        # Delete redis and other services
        node_details_list = self.remote_list_nodes()
        for node_name, node_details in node_details_list.items():
            self.node_leave_cluster(node_name)

        # Delete cluster folder
        rmtree(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}"))
        self.remote_clean_jobs(1)
        self.remote_delete_master()
        logger.info_green(f"The cluster {cluster_name} has been deleted.")

    def node_join_cluster(self, node_join_info: dict):
        node_name = node_join_info["name"]
        cluster_details = self.cluster_details
        node_ip_address = node_join_info["public_ip_address"]
        # Create user account
        logger.info(f"Now is going to create an user account for maro working node {node_name}.")
        if "super_user" in node_join_info:
            super_user = node_join_info["super_user"]
        else:
            super_user = ""
        GrassOnPremisesExecutor.create_user(
            admin_username=super_user,
            maro_user=cluster_details["user"]["admin_username"],
            ip_address=node_ip_address,
            pubkey=cluster_details["user"]["admin_public_key"],
            ssh_port=cluster_details["connection"]["ssh"]["port"]
        )

        self._create_node_data(node_join_info)
        self._init_node(node_name)

    def _create_node_data(self, node_join_info: dict):
        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details["id"]
        node_name = node_join_info["name"]
        node_ip_address = node_join_info["public_ip_address"]

        # Get resources
        cpu = node_join_info["resources"]["cpu"]
        memory = node_join_info["resources"]["memory"]
        gpu = node_join_info["resources"]["gpu"]

        # Save details
        node_details = {
            "public_ip_address": node_ip_address,
            "private_ip_address": node_ip_address,
            "node_size": "",
            "resource_name": f"{cluster_id}-{node_name}-vm",
            "hostname": f"{cluster_id}-{node_name}-vm",
            "resources": {
                "cpu": cpu,
                "memory": memory,
                "gpu": gpu
            },
            "containers": {}
        }
        self.remote_create_node(
            node_name=node_name,
            node_details=node_details,
        )

    def _init_node(self, node_name: str):
        logger.info(f"Initiating node {node_name}.")

        # Load details
        cluster_details = self.cluster_details
        admin_username = cluster_details["user"]["admin_username"]
        node_details = self.remote_get_node(node_name=node_name)
        node_public_ip_address = node_details["public_ip_address"]
        ssh_port = cluster_details["connection"]["ssh"]["port"]

        # Make sure the node is able to connect
        self.retry_connection_and_set_ssh_port(node_ip_address=node_public_ip_address)

        # Create folders
        path_list = {
            GlobalPaths.MARO_GRASS_LIB,
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}",
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/data",
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/image_files",
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs",
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/schedules"
        }
        self._create_path_in_list(node_public_ip_address, path_list)

        # Copy required files
        FileSynchronizer.copy_files_to_node(
            local_path=GlobalPaths.MARO_GRASS_LIB,
            remote_dir=GlobalPaths.MARO_LIB,
            admin_username=admin_username, node_ip_address=node_public_ip_address, ssh_port=ssh_port
        )
        FileSynchronizer.copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}",
            remote_dir=GlobalPaths.MARO_CLUSTERS,
            admin_username=admin_username, node_ip_address=node_public_ip_address, ssh_port=ssh_port
        )

        # Remote init node
        self.remote_init_node(
            node_name=node_name,
            node_ip_address=node_public_ip_address
        )

        # Save details
        self.remote_create_node(
            node_name=node_name,
            node_details=node_details
        )

        # Update node status
        # Since On-Premises machines don't need to shutdown, it will be set to start directly.
        self.remote_update_node_status(
            node_name=node_name,
            action="start"
        )

        # Load images
        self.remote_load_images(
            node_name=node_name,
            parallels=GlobalParams.PARALLELS,
            node_ip_address=node_public_ip_address
        )

        # Load node agent service
        self.remote_start_node_services(
            node_name=node_name,
            node_ip_address=node_public_ip_address
        )

        logger.info_green(f"Node {node_name} has been initialized.")

    def node_leave_cluster(self, node_name: str):
        cluster_details = self.cluster_details
        nodes_details = self.remote_list_nodes()
        if node_name not in nodes_details:
            logger.warning(f"The specified node cannot be found in cluster {cluster_details['name']}.")
            return

        node_details = nodes_details[node_name]
        # Update node status
        self.remote_update_node_status(
            node_name=node_name,
            action="stop"
        )
        # Delete node record in redis.
        self.remote_update_node_status(node_name, "delete")

        admin_username = cluster_details["user"]["admin_username"]
        node_ip_address = node_details["public_ip_address"]
        ssh_port = cluster_details["connection"]["ssh"]["port"]
        GrassOnPremisesExecutor.delete_user(
            admin_username="",
            maro_user=admin_username,
            ip_address=node_ip_address,
            ssh_port=ssh_port
        )
        logger.info_green(f"The node {node_name} has been left cluster {cluster_details['name']}.")

    @staticmethod
    def create_user(admin_username: str, maro_user: str, ip_address: str, pubkey: str, ssh_port: int) -> None:
        if "" == admin_username:
            print("Please input a user account that has permissions to create user:")
            admin_username = input("> ")

        FileSynchronizer.copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_GRASS_LIB}/scripts/create_user.py",
            remote_dir="~/",
            admin_username=admin_username, node_ip_address=ip_address, ssh_port=ssh_port
        )
        GrassOnPremisesExecutor.remote_add_user_to_node(admin_username, maro_user, ip_address, pubkey)

    @staticmethod
    def delete_user(admin_username: str, maro_user: str, ip_address: str, ssh_port: int) -> None:
        if "" == admin_username:
            admin_username = input("Please input a user account that has permissions to delete user:\r\n")

        FileSynchronizer.copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_GRASS_LIB}/scripts/delete_user.py",
            remote_dir="~/",
            admin_username=admin_username, node_ip_address=ip_address, ssh_port=ssh_port
        )

        GrassOnPremisesExecutor.remote_delete_user_from_node(admin_username, maro_user, ip_address)

    # Remote utils

    # Create a new user account on target OS.
    @staticmethod
    def remote_add_user_to_node(admin_username: str, maro_user: str, node_ip_address: str, pubkey: str):
        # The admin_user is an already exist account which has privileges to create new account on target OS.
        command = (
            f"ssh {admin_username}@{node_ip_address} 'sudo python3 ~/create_user.py {maro_user} \"{pubkey}\"'"
        )
        _ = SubProcess.run(command)

    # Delete maro cluster user account on target OS.
    @staticmethod
    def remote_delete_user_from_node(admin_username: str, delete_user: str, node_ip_address: str):
        # The admin_user is an already exist account which has privileges to create new account on target OS.
        command = (
            f"ssh {admin_username}@{node_ip_address} 'sudo python3 ~/delete_user.py {delete_user}'"
        )
        _ = SubProcess.run(command)
