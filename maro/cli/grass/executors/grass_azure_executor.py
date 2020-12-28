# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import collections
import json
import os
import secrets
import shutil
import string
import threading
import time
from copy import deepcopy
from multiprocessing.pool import ThreadPool

import yaml

from maro.cli.grass.executors.grass_executor import GrassExecutor
from maro.cli.grass.utils.copy import copy_and_rename, copy_files_from_node, copy_files_to_node
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
from maro.utils.exception.cli_exception import BadRequestError, CommandExecutionError, FileOperationError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class GrassAzureExecutor:

    def __init__(self, cluster_name: str):
        self.cluster_name = cluster_name
        self.cluster_details = load_cluster_details(cluster_name=cluster_name)
        self.grass_executor = GrassExecutor(cluster_details=self.cluster_details)

    # maro grass create

    @staticmethod
    def build_cluster_details(create_deployment: dict):
        # Standardize create deployment
        GrassAzureExecutor._standardize_create_deployment(create_deployment=create_deployment)

        # Get cluster name and save details
        cluster_name = create_deployment["name"]
        if os.path.isdir(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}"):
            raise BadRequestError(f"Cluster '{cluster_name}' is exist.")
        os.makedirs(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}")
        save_cluster_details(
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
        with open(f"{GlobalPaths.ABS_MARO_GRASS_LIB}/deployments/internal/grass_azure_create.yml") as fr:
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
            self._create_resource_group()
            self._create_vnet()
            # Simultaneously capture image and init master
            build_node_image_thread = threading.Thread(target=self._build_node_image, args=())
            build_node_image_thread.start()
            create_and_init_master_thread = threading.Thread(target=self._create_and_init_master, args=())
            create_and_init_master_thread.start()
            build_node_image_thread.join()
            create_and_init_master_thread.join()
        except Exception as e:
            # If failed, remove details folder, then raise
            shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}")
            raise e

        logger.info_green(f"Cluster {self.cluster_name} is created")

    def _set_cluster_id(self):
        # Set cluster id
        self.cluster_details["id"] = generate_cluster_id()

        # Save details
        save_cluster_details(
            cluster_name=self.cluster_name,
            cluster_details=self.cluster_details
        )

    def _create_resource_group(self):
        # Load and reload details
        subscription = self.cluster_details["cloud"]["subscription"]
        resource_group = self.cluster_details["cloud"]["resource_group"]
        location = self.cluster_details["cloud"]["location"]

        # Check if Azure CLI is installed
        version_details = AzureExecutor.get_version()
        logger.info_green(f"Your Azure CLI version: {version_details['azure-cli']}")

        # Set subscription id
        AzureExecutor.set_subscription(subscription=subscription)
        logger.info_green(f"Set subscription to: {subscription}")

        # Check and create resource group
        resource_group_details = AzureExecutor.get_resource_group(resource_group=resource_group)
        if resource_group_details is not None:
            logger.warning_yellow(f"Azure resource group {resource_group} already exists")
        else:
            AzureExecutor.create_resource_group(
                resource_group=resource_group,
                location=location
            )
            logger.info_green(f"Resource group: {resource_group} is created")

    def _create_vnet(self):
        logger.info("Creating vnet")

        # Load details
        resource_group = self.cluster_details["cloud"]["resource_group"]

        # Create ARM parameters and start deployment
        abs_template_file_path = f"{GlobalPaths.ABS_MARO_GRASS_LIB}/azure/create_vnet/template.json"
        abs_parameters_file_path = (
            f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/azure/create_vnet/parameters.json"
        )
        ArmTemplateParameterBuilder.create_vnet(
            cluster_details=self.cluster_details,
            export_path=abs_parameters_file_path
        )
        AzureExecutor.start_deployment(
            resource_group=resource_group,
            deployment_name="vnet",
            template_file_path=abs_template_file_path,
            parameters_file_path=abs_parameters_file_path
        )

        logger.info_green("Vnet is created")

    def _build_node_image(self):
        logger.info("Building MARO Node image")

        # Load details
        resource_name = "build-node-image"
        cluster_id = self.cluster_details["id"]
        resource_group = self.cluster_details["cloud"]["resource_group"]
        admin_username = self.cluster_details["user"]["admin_username"]
        ssh_port = self.cluster_details["connection"]["ssh"]["port"]
        image_name = f"{cluster_id}-node-image"
        vm_name = f"{cluster_id}-{resource_name}-vm"

        # Create ARM parameters and start deployment
        template_file_path = f"{GlobalPaths.ABS_MARO_GRASS_LIB}/azure/create_build_node_image_vm/template.json"
        parameters_file_path = (
            f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/azure/create_build_node_image_vm/parameters.json"
        )
        ArmTemplateParameterBuilder.create_build_node_image_vm(
            cluster_details=self.cluster_details,
            node_size="Standard_D4_v3",
            export_path=parameters_file_path
        )
        AzureExecutor.start_deployment(
            resource_group=resource_group,
            deployment_name=resource_name,
            template_file_path=template_file_path,
            parameters_file_path=parameters_file_path
        )
        # Gracefully wait
        time.sleep(10)

        # Get IP addresses
        ip_addresses = AzureExecutor.list_ip_addresses(
            resource_group=resource_group,
            vm_name=vm_name
        )
        public_ip_address = ip_addresses[0]["virtualMachine"]["network"]["publicIpAddresses"][0]["ipAddress"]

        # Make sure capture-node-image-vm is able to connect
        self.grass_executor.retry_connection_and_set_ssh_port(node_ip_address=public_ip_address)

        # Run init image script
        self._sync_mkdir(path=GlobalPaths.MARO_LOCAL_TMP, node_ip_address=public_ip_address)
        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_GRASS_LIB}/scripts/init_build_node_image_vm.py",
            remote_dir="~/",
            admin_username=admin_username, node_ip_address=public_ip_address, ssh_port=ssh_port
        )
        self.grass_executor.remote_init_build_node_image_vm(vm_ip_address=public_ip_address)

        # Extract image
        AzureExecutor.deallocate_vm(resource_group=resource_group, vm_name=vm_name)
        AzureExecutor.generalize_vm(resource_group=resource_group, vm_name=vm_name)
        AzureExecutor.create_image_from_vm(resource_group=resource_group, image_name=image_name, vm_name=vm_name)

        # Delete resources
        self._delete_resources(resource_name=resource_name)

        logger.info_green("MARO Node Image is built")

    def _create_and_init_master(self):
        logger.info("Creating MARO Master")
        self._create_master()
        self._init_master()
        logger.info_green("MARO Master is created")

    def _create_master(self):
        logger.info("Creating Master VM")

        # Load details
        master_details = self.cluster_details["master"]
        cluster_id = self.cluster_details["id"]
        resource_group = self.cluster_details["cloud"]["resource_group"]
        admin_username = self.cluster_details["user"]["admin_username"]
        node_size = self.cluster_details["master"]["node_size"]

        # Create ARM parameters and start deployment
        template_file_path = f"{GlobalPaths.ABS_MARO_GRASS_LIB}/azure/create_master/template.json"
        parameters_file_path = (
            f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/azure/create_master/parameters.json"
        )
        ArmTemplateParameterBuilder.create_master(
            cluster_details=self.cluster_details,
            node_size=node_size,
            export_path=parameters_file_path
        )
        AzureExecutor.start_deployment(
            resource_group=resource_group,
            deployment_name="master",
            template_file_path=template_file_path,
            parameters_file_path=parameters_file_path
        )

        # Get master IP addresses
        ip_addresses = AzureExecutor.list_ip_addresses(
            resource_group=resource_group,
            vm_name=f"{cluster_id}-master-vm"
        )
        public_ip_address = ip_addresses[0]["virtualMachine"]["network"]["publicIpAddresses"][0]["ipAddress"]
        private_ip_address = ip_addresses[0]["virtualMachine"]["network"]["privateIpAddresses"][0]
        hostname = f"{cluster_id}-master-vm"
        master_details["public_ip_address"] = public_ip_address
        master_details["private_ip_address"] = private_ip_address
        master_details["hostname"] = hostname
        master_details["resource_name"] = f"{cluster_id}-master-vm"
        logger.info_green(f"You can login to your master node with: ssh {admin_username}@{public_ip_address}")

        # Save details
        save_cluster_details(
            cluster_name=self.cluster_name,
            cluster_details=self.cluster_details,
            sync=False
        )

        logger.info_green("Master VM is created")

    def _init_master(self):
        logger.info("Initializing Master VM")

        # Load details
        master_details = self.cluster_details["master"]
        admin_username = self.cluster_details["user"]["admin_username"]
        master_public_ip_address = self.cluster_details["master"]["public_ip_address"]
        ssh_port = self.cluster_details["connection"]["ssh"]["port"]

        # Make sure master is able to connect
        self.grass_executor.retry_connection_and_set_ssh_port(node_ip_address=master_public_ip_address)

        # Create folders
        self._sync_mkdir(path=GlobalPaths.MARO_GRASS_LIB, node_ip_address=master_public_ip_address)
        self._sync_mkdir(
            path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}",
            node_ip_address=master_public_ip_address
        )
        self._sync_mkdir(
            path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/data",
            node_ip_address=master_public_ip_address
        )
        self._sync_mkdir(
            path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/images",
            node_ip_address=master_public_ip_address
        )
        self._sync_mkdir(
            path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs",
            node_ip_address=master_public_ip_address
        )
        self._sync_mkdir(
            path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/schedules",
            node_ip_address=master_public_ip_address
        )
        self._sync_mkdir(path=GlobalPaths.MARO_LOCAL_TMP, node_ip_address=master_public_ip_address)

        # Copy required files
        copy_files_to_node(
            local_path=GlobalPaths.MARO_GRASS_LIB,
            remote_dir=GlobalPaths.MARO_LIB,
            admin_username=admin_username, node_ip_address=master_public_ip_address, ssh_port=ssh_port
        )
        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}",
            remote_dir=GlobalPaths.MARO_CLUSTERS,
            admin_username=admin_username, node_ip_address=master_public_ip_address, ssh_port=ssh_port
        )

        # Get public key
        public_key = self.grass_executor.remote_get_public_key(node_ip_address=master_public_ip_address)

        # Remote init master
        self.grass_executor.remote_init_master()

        # Load master agent service
        self.grass_executor.remote_load_master_agent_service()

        # Save details
        master_details["public_key"] = public_key
        master_details["image_files"] = {}
        save_cluster_details(
            cluster_name=self.cluster_name,
            cluster_details=self.cluster_details
        )
        self.grass_executor.remote_set_master_details(master_details=master_details)

        logger.info_green("Master VM is initialized")

    # maro grass delete

    def delete(self):
        # Load details
        cluster_id = self.cluster_details["id"]
        resource_group = self.cluster_details["cloud"]["resource_group"]

        logger.info(f"Deleting cluster {self.cluster_name}")

        # Get resource list
        resource_list = AzureExecutor.list_resources(resource_group=resource_group)

        # Filter resources
        deletable_ids = []
        for resource_info in resource_list:
            if resource_info["name"].startswith(cluster_id):
                deletable_ids.append(resource_info["id"])

        # Delete resources
        if len(deletable_ids) > 0:
            AzureExecutor.delete_resources(resources=deletable_ids)

        # Delete cluster folder
        shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}")

        logger.info_green(f"Cluster {self.cluster_name} is deleted")

    # maro grass node

    def scale_node(self, replicas: int, node_size: str):
        # Load details
        nodes_details = self.grass_executor.remote_get_nodes_details()

        # Init node_size_to_count
        node_size_to_count = collections.defaultdict(lambda: 0)
        for node_name, node_details in nodes_details.items():
            node_size_to_count[node_details["node_size"]] += 1

        # Get node_size_to_spec
        node_size_to_spec = self._get_node_size_to_spec()
        if node_size not in node_size_to_spec:
            raise BadRequestError(f"Invalid node_size '{node_size}'.")

        # Scale nodes
        if node_size_to_count[node_size] > replicas:
            self._delete_nodes(
                num=node_size_to_count[node_size] - replicas,
                node_size=node_size
            )
        elif node_size_to_count[node_size] < replicas:
            self._create_nodes(
                num=replicas - node_size_to_count[node_size],
                node_size=node_size,
                node_size_to_spec=node_size_to_spec
            )
        else:
            logger.warning_yellow("Replica is match, no create or delete")

    def _create_nodes(self, num: int, node_size: str, node_size_to_spec: dict) -> None:
        logger.info(f"Scaling up {num}")

        # Parallel create
        with ThreadPool(GlobalParams.PARALLELS) as pool:
            pool.starmap(
                self._create_node,
                [[node_size, node_size_to_spec]] * num
            )

    def _create_node(self, node_size: str, node_size_to_spec: dict):
        # Generate node name
        node_name = generate_node_name()
        logger.info(message=f"Creating node {node_name}")

        # Create node
        self._create_vm(
            node_name=node_name,
            node_size=node_size,
            node_size_to_spec=node_size_to_spec
        )

        # Init node
        self._init_node(
            node_name=node_name
        )

        logger.info_green(message=f"Node {node_name} is created")

    def _delete_nodes(self, num: int, node_size: str) -> None:
        # Load details
        nodes_details = self.grass_executor.remote_get_nodes_details()

        # Get deletable_nodes and check, TODO: consider to add -f
        deletable_nodes = []
        for node_name, node_details in nodes_details.items():
            if node_details["node_size"] == node_size and len(node_details["containers"]) == 0:
                deletable_nodes.append(node_name)
        if len(deletable_nodes) >= num:
            logger.info(f"Scaling down {num}")

            # Parallel delete
            params = [[deletable_node] for deletable_node in deletable_nodes[:num]]
            with ThreadPool(GlobalParams.PARALLELS) as pool:
                pool.starmap(
                    self._delete_node,
                    params
                )
        else:
            logger.warning_yellow(
                "Unable to scale down."
                f" Only {len(deletable_nodes)} are deletable, but need to delete {num} to meet the replica"
            )

    def _create_vm(self, node_name: str, node_size: str, node_size_to_spec: dict):
        logger.info(message=f"Creating VM {node_name}")

        # Load details
        location = self.cluster_details["cloud"]["location"]
        cluster_id = self.cluster_details["id"]
        resource_group = self.cluster_details["cloud"]["resource_group"]
        image_name = f"{cluster_id}-node-image"
        image_resource_id = AzureExecutor.get_image_resource_id(resource_group=resource_group, image_name=image_name)

        # Create ARM parameters and start deployment
        template_file_path = f"{GlobalPaths.ABS_MARO_GRASS_LIB}/azure/create_node/template.json"
        parameters_file_path = (
            f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/azure/create_{node_name}/parameters.json"
        )
        ArmTemplateParameterBuilder.create_node(
            node_name=node_name,
            cluster_details=self.cluster_details,
            node_size=node_size,
            image_resource_id=image_resource_id,
            export_path=parameters_file_path
        )
        AzureExecutor.start_deployment(
            resource_group=resource_group,
            deployment_name=node_name,
            template_file_path=template_file_path,
            parameters_file_path=parameters_file_path
        )

        # Get node IP addresses
        ip_addresses = AzureExecutor.list_ip_addresses(
            resource_group=resource_group,
            vm_name=f"{cluster_id}-{node_name}-vm"
        )

        # Get sku and check gpu nums
        gpu_nums = 0
        node_size_sku = AzureExecutor.get_sku(
            vm_size=node_size, location=location)
        if node_size_sku is not None:
            for capability in node_size_sku["capabilities"]:
                if capability["name"] == "GPUs":
                    gpu_nums = int(capability["value"])
                    break

        # Save details
        node_details = {
            "name": node_name,
            "id": node_name,
            "public_ip_address": ip_addresses[0]["virtualMachine"]["network"]["publicIpAddresses"][0]["ipAddress"],
            "private_ip_address": ip_addresses[0]["virtualMachine"]["network"]["privateIpAddresses"][0],
            "node_size": node_size,
            "resource_name": f"{cluster_id}-{node_name}-vm",
            "hostname": f"{cluster_id}-{node_name}-vm",
            "resources": {
                "cpu": node_size_to_spec[node_size]["numberOfCores"],
                "memory": node_size_to_spec[node_size]["memoryInMb"],
                "gpu": gpu_nums
            },
            "containers": {}
        }
        self.grass_executor.remote_set_node_details(
            node_name=node_name,
            node_details=node_details,
        )

        logger.info_green(f"VM {node_name} is created")

    def _delete_node(self, node_name: str):
        logger.info(f"Deleting node {node_name}")

        # Load details
        resource_group = self.cluster_details["cloud"]["resource_group"]

        # Delete resources
        self._delete_resources(resource_name=node_name)

        # Delete azure deployment
        AzureExecutor.delete_deployment(
            resource_group=resource_group,
            deployment_name=node_name
        )

        # Delete parameters_file
        shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/azure/create_{node_name}")

        # Update node status
        self.grass_executor.remote_update_node_status(
            node_name=node_name,
            action="delete"
        )

        logger.info_green(f"Node {node_name} is deleted")

    def _init_node(self, node_name: str):
        logger.info(f"Initiating node {node_name}")

        # Load details
        admin_username = self.cluster_details["user"]["admin_username"]
        node_details = self.grass_executor.remote_get_node_details(node_name=node_name)
        node_public_ip_address = node_details["public_ip_address"]
        ssh_port = self.cluster_details["connection"]["ssh"]["port"]

        # Make sure the node is able to connect
        self.grass_executor.retry_connection_and_set_ssh_port(node_ip_address=node_public_ip_address)

        # Copy required files
        self._sync_mkdir(path=f"{GlobalPaths.MARO_LOCAL_TMP}", node_ip_address=node_public_ip_address)
        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_GRASS_LIB}/scripts/init_node.py",
            remote_dir="~/",
            admin_username=admin_username, node_ip_address=node_public_ip_address, ssh_port=ssh_port
        )
        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/details.yml",
            remote_dir="~/",
            admin_username=admin_username, node_ip_address=node_public_ip_address, ssh_port=ssh_port
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
        self.grass_executor.remote_update_node_status(
            node_name=node_name,
            action="create"
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

        logger.info_green(f"Node {node_name} is initialized")

    def start_node(self, replicas: int, node_size: str):
        # Get nodes details
        nodes_details = self.grass_executor.remote_get_nodes_details()

        # Get startable nodes
        startable_nodes = []
        for node_name, node_details in nodes_details.items():
            if node_details["node_size"] == node_size and node_details["state"] == "Stopped":
                startable_nodes.append(node_name)

        # Check replicas
        if len(startable_nodes) < replicas:
            raise BadRequestError(
                f"No enough '{node_size}' nodes can be started (only {len(startable_nodes)} is startable)."
            )

        # Parallel start
        params = [[startable_node] for startable_node in startable_nodes[:replicas]]
        with ThreadPool(GlobalParams.PARALLELS) as pool:
            pool.starmap(
                self._start_node,
                params
            )

    def _start_node(self, node_name: str):
        logger.info(f"Starting node {node_name}")

        # Load details
        cluster_id = self.cluster_details["id"]
        resource_group = self.cluster_details["cloud"]["resource_group"]
        node_details = self.grass_executor.remote_get_node_details(node_name=node_name)
        node_public_ip_address = node_details["public_ip_address"]

        # Start node
        AzureExecutor.start_vm(
            resource_group=resource_group,
            vm_name=f"{cluster_id}-{node_name}-vm"
        )

        # Update node status
        self.grass_executor.remote_update_node_status(
            node_name=node_name,
            action="start"
        )

        # Make sure the node is able to connect
        self.grass_executor.retry_connection_and_set_ssh_port(
            node_ip_address=node_public_ip_address
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

        logger.info_green(f"Node {node_name} is started")

    def stop_node(self, replicas: int, node_size: str):
        # Get nodes details
        nodes_details = self.grass_executor.remote_get_nodes_details()

        # Get stoppable nodes
        stoppable_nodes = []
        for node_name, node_details in nodes_details.items():
            if (
                node_details["node_size"] == node_size and
                node_details["state"] == "Running" and
                self._count_running_containers(node_details) == 0
            ):
                stoppable_nodes.append(node_name)

        # Check replicas
        if len(stoppable_nodes) < replicas:
            raise BadRequestError(
                f"No more '{node_size}' nodes can be stopped, only {len(stoppable_nodes)} are stoppable."
            )

        # Parallel stop
        params = [[stoppable_node] for stoppable_node in stoppable_nodes[:replicas]]
        with ThreadPool(GlobalParams.PARALLELS) as pool:
            pool.starmap(
                self._stop_node,
                params
            )

    def _stop_node(self, node_name: str):
        logger.info(f"Stopping node {node_name}")

        # Load details
        cluster_id = self.cluster_details["id"]
        resource_group = self.cluster_details["cloud"]["resource_group"]

        # Stop node
        AzureExecutor.stop_vm(
            resource_group=resource_group,
            vm_name=f"{cluster_id}-{node_name}-vm"
        )

        # Update node status
        self.grass_executor.remote_update_node_status(
            node_name=node_name,
            action="stop"
        )

        logger.info_green(f"Node {node_name} is stopped")

    def _get_node_size_to_spec(self) -> dict:
        # Load details
        location = self.cluster_details["cloud"]["location"]

        # List available sizes for VMs
        specs = AzureExecutor.list_vm_sizes(location=location)

        # Get node_size_to_spec
        node_size_to_spec = {}
        for spec in specs:
            node_size_to_spec[spec["name"]] = spec

        return node_size_to_spec

    def list_node(self):
        # Get nodes details
        nodes_details = self.grass_executor.remote_get_nodes_details()

        # Print details
        logger.info(
            json.dumps(
                nodes_details,
                indent=4, sort_keys=True
            )
        )

    @staticmethod
    def _count_running_containers(node_details: dict):
        # Extract details
        containers_details = node_details["containers"]

        # Do counting
        count = 0
        for container_details in containers_details:
            if container_details["Status"] == "running":
                count += 1

        return count

    # maro grass image

    def push_image(
        self, image_name: str, image_path: str, remote_context_path: str,
        remote_image_name: str
    ):
        # Load details
        admin_username = self.cluster_details["user"]["admin_username"]
        master_public_ip_address = self.cluster_details["master"]["public_ip_address"]
        ssh_port = self.cluster_details["connection"]["ssh"]["port"]

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
                admin_username=admin_username, node_ip_address=master_public_ip_address, ssh_port=ssh_port
            )
            self.grass_executor.remote_update_image_files_details()
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
                admin_username=admin_username, node_ip_address=master_public_ip_address, ssh_port=ssh_port
            )
            self.grass_executor.remote_update_image_files_details()
            self._batch_load_images()
        elif remote_context_path and remote_image_name:
            self.grass_executor.remote_build_image(
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
        nodes_details = self.grass_executor.remote_get_nodes_details()

        # build params
        params = []
        for node_name, node_details in nodes_details.items():
            if node_details["state"] == "Running":
                params.append([
                    node_name,
                    GlobalParams.PARALLELS,
                    node_details["public_ip_address"]
                ])

        # Parallel load image
        with ThreadPool(GlobalParams.PARALLELS) as pool:
            pool.starmap(
                self._load_image,
                params
            )

    def _load_image(self, node_name: str, parallels: int, node_ip_address: str):
        self.grass_executor.remote_load_images(
            node_name=node_name,
            parallels=parallels,
            node_ip_address=node_ip_address
        )

    def _check_checksum_validity(self, local_file_path: str, remote_file_path: str) -> bool:
        local_checksum = get_checksum(file_path=local_file_path)
        remote_checksum = self.grass_executor.remote_get_checksum(
            file_path=remote_file_path
        )
        return local_checksum == remote_checksum

    # maro grass data

    def push_data(self, local_path: str, remote_path: str):
        # Load details
        admin_username = self.cluster_details["user"]["admin_username"]
        master_public_ip_address = self.cluster_details["master"]["public_ip_address"]
        ssh_port = self.cluster_details["connection"]["ssh"]["port"]

        if not remote_path.startswith("/"):
            raise FileOperationError(f"Invalid remote path: {remote_path}\nShould be started with '/'.")
        copy_files_to_node(
            local_path=local_path,
            remote_dir=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/data{remote_path}",
            admin_username=admin_username, node_ip_address=master_public_ip_address, ssh_port=ssh_port
        )

    def pull_data(self, local_path: str, remote_path: str):
        # Load details
        admin_username = self.cluster_details["user"]["admin_username"]
        master_public_ip_address = self.cluster_details["master"]["public_ip_address"]
        ssh_port = self.cluster_details["connection"]["ssh"]["port"]

        if not remote_path.startswith("/"):
            raise FileOperationError(f"Invalid remote path: {remote_path}\nShould be started with '/'.")
        copy_files_from_node(
            local_dir=local_path,
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/data{remote_path}",
            admin_username=admin_username, node_ip_address=master_public_ip_address, ssh_port=ssh_port
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
        logger.info(f"Start sending job ticket {job_details['name']}")

        # Load details
        admin_username = self.cluster_details["user"]["admin_username"]
        master_public_ip_address = self.cluster_details["master"]["public_ip_address"]
        ssh_port = self.cluster_details["connection"]["ssh"]["port"]
        job_name = job_details["name"]

        # Sync mkdir
        self._sync_mkdir(
            path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs/{job_name}",
            node_ip_address=master_public_ip_address
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
            admin_username=admin_username, node_ip_address=master_public_ip_address, ssh_port=ssh_port
        )

        # Remote start job
        self.grass_executor.remote_create_job_details(job_name=job_name)
        self.grass_executor.remote_create_pending_job_ticket(job_name=job_name)

        logger.info_green(f"Job ticket {job_details['name']} is sent")

    def stop_job(self, job_name: str):
        # Remote stop job
        self.grass_executor.remote_create_killed_job_ticket(job_name=job_name)
        self.grass_executor.remote_delete_pending_job_ticket(job_name=job_name)

    def list_job(self):
        # Get jobs details
        jobs_details = self.grass_executor.remote_get_jobs_details()

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
        admin_username = self.cluster_details["user"]["admin_username"]
        master_public_ip_address = self.cluster_details["master"]["public_ip_address"]
        ssh_port = self.cluster_details["connection"]["ssh"]["port"]
        job_id = job_details["id"]

        # Copy logs from master
        try:
            copy_files_from_node(
                local_dir=export_dir,
                remote_path=f"~/.maro/logs/{job_id}",
                admin_username=admin_username, node_ip_address=master_public_ip_address, ssh_port=ssh_port
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

        # Load details
        master_public_ip_address = self.cluster_details["master"]["public_ip_address"]

        # Sync mkdir
        self._sync_mkdir(
            path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/schedules/{schedule_name}",
            node_ip_address=master_public_ip_address
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
                self.grass_executor.remote_create_killed_job_ticket(job_name=job_name)
                self.grass_executor.remote_delete_pending_job_ticket(job_name=job_name)

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

        job_details = deepcopy(schedule_details)
        job_details["name"] = job_name
        job_details["tags"] = {
            "schedule": schedule_name
        }
        job_details.pop("job_names")

        return job_details

    # maro grass clean

    def clean(self):
        # TODO add clean redis

        # Remote clean
        self.grass_executor.remote_clean(parallels=GlobalParams.PARALLELS)

    # maro grass status

    def status(self, resource_name: str):
        if resource_name == "master":
            return_status = self.grass_executor.remote_get_master_details()
        elif resource_name == "nodes":
            return_status = self.grass_executor.remote_get_nodes_details()
        elif resource_name == "containers":
            return_status = self.grass_executor.remote_get_containers_details()
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

    # Utils

    def _delete_resources(self, resource_name: str):
        # Get params
        cluster_id = self.cluster_details["id"]
        resource_group = self.cluster_details["cloud"]["resource_group"]

        # Get resource list
        resource_list = AzureExecutor.list_resources(resource_group=resource_group)

        # Filter resources
        deletable_ids = []
        for resource_info in resource_list:
            if resource_info["name"].startswith(f"{cluster_id}-{resource_name}"):
                deletable_ids.append(resource_info["id"])

        # Delete resources
        if len(deletable_ids) > 0:
            AzureExecutor.delete_resources(resources=deletable_ids)

    def _sync_mkdir(self, path: str, node_ip_address: str):
        """Mkdir synchronously at local and remote.

        Args:
            path (str): path of the file, should be a string with an initial component of ~ or ~user
            node_ip_address (str): ip address of the remote node
        """
        # Create local dir
        os.makedirs(os.path.expanduser(path), exist_ok=True)

        # Create remote dir
        self.grass_executor.remote_mkdir(node_ip_address=node_ip_address, path=path)


class ArmTemplateParameterBuilder:
    @staticmethod
    def create_vnet(cluster_details: dict, export_path: str) -> dict:
        # Get params
        cluster_id = cluster_details["id"]
        location = cluster_details["cloud"]["location"]

        # Load and update parameters
        with open(f"{GlobalPaths.ABS_MARO_GRASS_LIB}/azure/create_vnet/parameters.json", "r") as f:
            base_parameters = json.load(f)
            parameters = base_parameters["parameters"]
            parameters["location"]["value"] = location
            parameters["virtualNetworkName"]["value"] = f"{cluster_id}-vnet"

        # Export parameters if the path is set
        if export_path:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w") as fw:
                json.dump(base_parameters, fw, indent=4)

        return base_parameters

    @staticmethod
    def create_master(cluster_details: dict, node_size: str, export_path: str) -> dict:
        # Get params
        resource_name = "master"
        cluster_id = cluster_details["id"]
        location = cluster_details["cloud"]["location"]
        admin_username = cluster_details["user"]["admin_username"]
        admin_public_key = cluster_details["user"]["admin_public_key"]
        ssh_port = cluster_details["connection"]["ssh"]["port"]

        # Load and update parameters
        with open(f"{GlobalPaths.ABS_MARO_GRASS_LIB}/azure/create_master/parameters.json", "r") as f:
            base_parameters = json.load(f)
            parameters = base_parameters["parameters"]
            parameters["location"]["value"] = location
            parameters["networkInterfaceName"]["value"] = f"{cluster_id}-{resource_name}-nic"
            parameters["networkSecurityGroupName"]["value"] = f"{cluster_id}-{resource_name}-nsg"
            parameters["virtualNetworkName"]["value"] = f"{cluster_id}-vnet"
            parameters["publicIpAddressName"]["value"] = f"{cluster_id}-{resource_name}-pip"
            parameters["virtualMachineName"]["value"] = f"{cluster_id}-{resource_name}-vm"
            parameters["virtualMachineSize"]["value"] = node_size
            parameters["adminUsername"]["value"] = admin_username
            parameters["adminPublicKey"]["value"] = admin_public_key
            parameters["sshDestinationPort"]["value"] = f"{ssh_port}"

        # Export parameters if the path is set
        if export_path:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w") as fw:
                json.dump(base_parameters, fw, indent=4)

        return base_parameters

    @staticmethod
    def create_build_node_image_vm(cluster_details: dict, node_size: str, export_path: str) -> dict:
        # Get params
        resource_name = "build-node-image"
        cluster_id = cluster_details["id"]
        location = cluster_details["cloud"]["location"]
        admin_username = cluster_details["user"]["admin_username"]
        admin_public_key = cluster_details["user"]["admin_public_key"]
        ssh_port = cluster_details["connection"]["ssh"]["port"]

        # Load and update parameters
        with open(f"{GlobalPaths.ABS_MARO_GRASS_LIB}/azure/create_build_node_image_vm/parameters.json", "r") as f:
            base_parameters = json.load(f)
            parameters = base_parameters["parameters"]
            parameters["location"]["value"] = location
            parameters["networkInterfaceName"]["value"] = f"{cluster_id}-{resource_name}-nic"
            parameters["networkSecurityGroupName"]["value"] = f"{cluster_id}-{resource_name}-nsg"
            parameters["virtualNetworkName"]["value"] = f"{cluster_id}-vnet"
            parameters["publicIpAddressName"]["value"] = f"{cluster_id}-{resource_name}-pip"
            parameters["virtualMachineName"]["value"] = f"{cluster_id}-{resource_name}-vm"
            parameters["virtualMachineSize"]["value"] = node_size
            parameters["adminUsername"]["value"] = admin_username
            parameters["adminPublicKey"]["value"] = admin_public_key
            parameters["sshDestinationPort"]["value"] = f"{ssh_port}"

        # Export parameters if the path is set
        if export_path:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w") as fw:
                json.dump(base_parameters, fw, indent=4)

        return base_parameters

    @staticmethod
    def create_node(
        node_name: str, cluster_details: dict,
        node_size: str, image_resource_id: str,
        export_path: str
    ) -> dict:
        # Extract variables
        resource_name = node_name
        cluster_id = cluster_details["id"]
        location = cluster_details["cloud"]["location"]
        admin_username = cluster_details["user"]["admin_username"]
        admin_public_key = cluster_details["user"]["admin_public_key"]
        ssh_port = cluster_details["connection"]["ssh"]["port"]

        # Load and update parameters
        with open(f"{GlobalPaths.ABS_MARO_GRASS_LIB}/azure/create_node/parameters.json", "r") as f:
            base_parameters = json.load(f)
            parameters = base_parameters["parameters"]
            parameters["location"]["value"] = location
            parameters["networkInterfaceName"]["value"] = f"{cluster_id}-{resource_name}-nic"
            parameters["networkSecurityGroupName"]["value"] = f"{cluster_id}-{resource_name}-nsg"
            parameters["virtualNetworkName"]["value"] = f"{cluster_id}-vnet"
            parameters["publicIpAddressName"]["value"] = f"{cluster_id}-{resource_name}-pip"
            parameters["virtualMachineName"]["value"] = f"{cluster_id}-{resource_name}-vm"
            parameters["virtualMachineSize"]["value"] = node_size
            parameters["imageResourceId"]["value"] = image_resource_id
            parameters["adminUsername"]["value"] = admin_username
            parameters["adminPublicKey"]["value"] = admin_public_key
            parameters["sshDestinationPort"]["value"] = f"{ssh_port}"

        # Export parameters if the path is set
        if export_path:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w") as fw:
                json.dump(base_parameters, fw, indent=4)

        return base_parameters
