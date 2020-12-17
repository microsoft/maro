# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import collections
import json
import os
import secrets
import string
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from shutil import rmtree

import yaml

from maro.cli.grass.executors.grass_executor import GrassExecutor
from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
from maro.cli.grass.utils.copy import copy_and_rename, copy_files_from_node, copy_files_to_node, sync_mkdir
from maro.cli.grass.utils.hash import get_checksum
from maro.cli.utils.details import (
    load_cluster_details, load_job_details, load_schedule_details, save_cluster_details, save_job_details,
    save_schedule_details
)
from maro.cli.utils.executors.azure_executor import AzureExecutor
from maro.cli.utils.naming import (
    generate_cluster_id, generate_component_id, generate_job_id, generate_node_name, get_valid_file_name
)
from maro.cli.utils.params import GlobalParams, GlobalPaths
from maro.cli.utils.subprocess import SubProcess
from maro.cli.utils.validation import validate_and_fill_dict
from maro.utils.exception.cli_exception import CliException, CommandError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class GrassOnPremisesExecutor:

    def __init__(self, cluster_name: str):
        self.cluster_name = cluster_name
        self.cluster_details = load_cluster_details(cluster_name=cluster_name)
        self.grass_executor = GrassExecutor(cluster_details=self.cluster_details)

    # maro grass create

    @staticmethod
    def build_cluster_details(create_deployment: dict):
        # Standardize create deployment
        GrassOnPremisesExecutor._standardize_create_deployment(create_deployment=create_deployment)

        # Create user account
        GrassOnPremisesExecutor.create_user(
            "",
            create_deployment["user"]["admin_username"],
            create_deployment["master"]["public_ip_address"],
            create_deployment["user"]["admin_public_key"])

        # Get cluster name and save details
        cluster_name = create_deployment["name"]
        if os.path.isdir(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}")):
            raise CliException(f"Cluster {cluster_name} already exist.")
        os.makedirs(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}"))
        save_cluster_details(
            cluster_name=cluster_name,
            cluster_details=create_deployment
        )

    @staticmethod
    def _standardize_create_deployment(create_deployment: dict):
        alphabet = string.ascii_letters + string.digits
        optional_key_to_value = {
            "root['master']['redis']": {'port': 6379},
            "root['master']['redis']['port']": 6379,
            "root['master']['fluentd']": {'port': 24224},
            "root['master']['fluentd']['port']": 24224,
            "root['master']['samba']": {'password': ''.join(secrets.choice(alphabet) for _ in range(20))},
            "root['master']['samba']['password']": ''.join(secrets.choice(alphabet) for _ in range(20))
        }
        with open(os.path.expanduser(
                f'{GlobalPaths.MARO_GRASS_LIB}/deployments/internal/grass-on-premises-create.yml')) as fr:
            create_deployment_template = yaml.safe_load(fr)
        validate_and_fill_dict(
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
            raise e

        logger.info_green(f"Cluster {self.cluster_name} has been created.")

    def _set_cluster_id(self):
        # Load details
        cluster_details = self.cluster_details

        # Set cluster id
        cluster_details["id"] = generate_cluster_id()

        # Save details
        save_cluster_details(
            cluster_name=self.cluster_name,
            cluster_details=cluster_details
        )

    def _set_master_info(self):
        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details["id"]
        master_details = cluster_details["master"]
        #hostname = f"{cluster_id}-master-vm"
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

        # Make sure master is able to connect
        self.grass_executor.retry_until_connected(node_ip_address=master_public_ip_address)

        # Create folders
        sync_mkdir(
            remote_path=GlobalPaths.MARO_GRASS_LIB,
            admin_username=admin_username, node_ip_address=master_public_ip_address
        )
        sync_mkdir(
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}",
            admin_username=admin_username, node_ip_address=master_public_ip_address
        )
        sync_mkdir(
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/data",
            admin_username=admin_username, node_ip_address=master_public_ip_address
        )
        sync_mkdir(
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/images",
            admin_username=admin_username, node_ip_address=master_public_ip_address
        )
        sync_mkdir(
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs",
            admin_username=admin_username, node_ip_address=master_public_ip_address
        )
        sync_mkdir(
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/schedules",
            admin_username=admin_username, node_ip_address=master_public_ip_address
        )

        # Copy required files
        copy_files_to_node(
            local_path=GlobalPaths.MARO_GRASS_LIB,
            remote_dir=GlobalPaths.MARO_LIB,
            admin_username=admin_username, node_ip_address=master_public_ip_address
        )
        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}",
            remote_dir=GlobalPaths.MARO_CLUSTERS,
            admin_username=admin_username, node_ip_address=master_public_ip_address
        )

        # Get public key
        public_key = self.grass_executor.remote_get_public_key(node_ip_address=master_public_ip_address)

        # Remote init master
        self.grass_executor.remote_init_master()

        # Load master agent service
        self.grass_executor.remote_load_master_agent_service()

        # Save details
        master_details['public_key'] = public_key
        # master_details['image_files'] = {}
        save_cluster_details(
            cluster_name=self.cluster_name,
            cluster_details=cluster_details
        )
        self.grass_executor.remote_set_master_details(master_details=cluster_details['master'])

        logger.info_green("Master node is initialized")

    def delete(self):
        # Load details
        cluster_name = self.cluster_name
        cluster_details = load_cluster_details(cluster_name=cluster_name)
        cluster_id = cluster_details["id"]

        logger.info(f"Deleting cluster {cluster_name}")
        # Delete redis and other services

        # Delete cluster folder
        rmtree(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}"))

        logger.info_green(f"The cluster {cluster_name} has been deleted.")

    def node_join_cluster(self, node_join_info: dict):
        node_name = node_join_info["name"]
        cluster_details = self.cluster_details
        node_ip_address = node_join_info["public_ip_address"]
        # Create user account
        GrassOnPremisesExecutor.create_user(
            "",
            cluster_details["user"]["admin_username"],
            node_ip_address,
            cluster_details["user"]["admin_public_key"])

        self._create_node_data(node_join_info)
        self._init_node(node_name)


    def _create_node_data(self, node_join_info: dict):
        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details["id"]
        node_name = node_join_info["name"]
        master_details = cluster_details["master"]
        node_ip_address = node_join_info["public_ip_address"]

        # Get resources
        cpu = node_join_info["resources"]["cpu"]
        memory = node_join_info["resources"]["memory"]
        gpu = node_join_info["resources"]["gpu"]

        # Save details
        node_details = {
            'public_ip_address': node_ip_address,
            'private_ip_address': node_ip_address,
            'node_size': "",
            'resource_name': f"{cluster_id}-{node_name}-vm",
            'hostname': f"{cluster_id}-{node_name}-vm",
            'resources': {
                'cpu': cpu,
                'memory': memory,
                'gpu': gpu
            },
            'containers': {}
        }
        self.grass_executor.remote_set_node_details(
            node_name=node_name,
            node_details=node_details,
        )

    def _init_node(self, node_name: str):
        logger.info(f"Initiating node {node_name}.")

        # Load details
        cluster_details = self.cluster_details
        admin_username = cluster_details["user"]["admin_username"]
        node_details = self.grass_executor.remote_get_node_details(node_name=node_name)
        node_public_ip_address = node_details["public_ip_address"]

        # Make sure the node is able to connect
        self.grass_executor.retry_until_connected(node_ip_address=node_public_ip_address)

        # Create folders
        sync_mkdir(
            remote_path=GlobalPaths.MARO_GRASS_LIB,
            admin_username=admin_username, node_ip_address=node_public_ip_address
        )
        sync_mkdir(
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}",
            admin_username=admin_username, node_ip_address=node_public_ip_address
        )
        sync_mkdir(
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/data",
            admin_username=admin_username, node_ip_address=node_public_ip_address
        )
        sync_mkdir(
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/images",
            admin_username=admin_username, node_ip_address=node_public_ip_address
        )
        sync_mkdir(
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs",
            admin_username=admin_username, node_ip_address=node_public_ip_address
        )
        sync_mkdir(
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/schedules",
            admin_username=admin_username, node_ip_address=node_public_ip_address
        )

        # Copy required files
        copy_files_to_node(
            local_path=GlobalPaths.MARO_GRASS_LIB,
            remote_dir=GlobalPaths.MARO_LIB,
            admin_username=admin_username, node_ip_address=node_public_ip_address
        )
        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}",
            remote_dir=GlobalPaths.MARO_CLUSTERS,
            admin_username=admin_username, node_ip_address=node_public_ip_address
        )

        # Remote init node
        self.grass_executor.remote_init_node(
            node_name=node_name,
            node_ip_address=node_public_ip_address
        )

        # Get public key
        public_key = self.grass_executor.remote_get_public_key(node_ip_address=node_public_ip_address)

        # Save details
        node_details["public_key"] = public_key
        self.grass_executor.remote_set_node_details(
            node_name=node_name,
            node_details=node_details
        )

        # Update node status
        # Since On-Premises machines don't need to shutdown, it will be set to start directly.
        self.grass_executor.remote_update_node_status(
            node_name=node_name,
            action="start"
        )

        # Load images
        self.grass_executor.remote_load_images(
            node_name=node_name,
            parallels=GlobalParams.PARALLELS,
            node_ip_address=node_public_ip_address
        )

        # Load node agent service
        self.grass_executor.remote_load_node_agent_service(
            node_name=node_name,
            node_ip_address=node_public_ip_address
        )

        logger.info_green(f"Node {node_name} has been initialized.")

    def node_leave_cluster(self, node_name: str):
        pass

    @staticmethod
    def create_user(admin_username: str, maro_user: str, ip_address: str, pubkey: str) -> None:
        if("" == admin_username):
            admin_username = input("Please input a user account that has permissions to create user:\r\n")

        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_GRASS_LIB}/scripts/create_user.py",
            remote_dir="~/",
            admin_username=admin_username, node_ip_address=ip_address)
        GrassExecutor.remote_add_user_to_node(admin_username, maro_user, ip_address, pubkey)

    def delete_user(self, admin_username: str, ip_address: str) -> None:
        executor = self.grass_executor
        if("" == admin_username):
            admin_username = input("Please input a user account that has permissions to create user:\r\n")

        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_GRASS_LIB}/scripts/delete_user.py",
            remote_dir="~/",
            admin_username=admin_username, node_ip_address=ip_address)

        self.grass_executor.remote_delete_user_from_node(admin_username, ip_address)

    @staticmethod
    def is_legal_ip(test_str: str):

        if "." not in test_str:
            return False
        elif test_str.count(".") != 3:
            return False
        else:
            flag = True
            one_list = test_str.split(".")
            for one in one_list:
                try:
                    one_num = int(one)
                    if one_num >= 0 and one_num <= 255:
                        pass
                    else:
                        flag = False
                except:
                    flag = False
            return flag
