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
from multiprocessing.pool import ThreadPool

import yaml

from maro.cli.grass.executors.grass_executor import GrassExecutor
from maro.cli.grass.utils.copy import copy_files_to_node
from maro.cli.grass.utils.params import ContainerStatus, NodeStatus
from maro.cli.utils.details import load_cluster_details, save_cluster_details
from maro.cli.utils.executors.azure_executor import AzureExecutor
from maro.cli.utils.naming import generate_cluster_id, generate_node_name
from maro.cli.utils.params import GlobalParams, GlobalPaths
from maro.cli.utils.validation import validate_and_fill_dict
from maro.utils.exception.cli_exception import BadRequestError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class GrassAzureExecutor(GrassExecutor):

    def __init__(self, cluster_name: str):
        super().__init__(cluster_details=load_cluster_details(cluster_name=cluster_name))

        # Cloud configs
        self.subscription = self.cluster_details["cloud"]["subscription"]
        self.resource_group = self.cluster_details["cloud"]["resource_group"]
        self.location = self.cluster_details["cloud"]["location"]

    # maro grass create

    @staticmethod
    def build_cluster_details(create_deployment: dict):
        # Standardize create deployment
        GrassAzureExecutor._standardize_create_deployment(create_deployment=create_deployment)

        # Get cluster name and create folders
        cluster_name = create_deployment["name"]
        if os.path.isdir(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}"):
            raise BadRequestError(f"Cluster '{cluster_name}' is exist.")
        os.makedirs(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}")

        # Set params and save details
        create_deployment["id"] = generate_cluster_id()
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

    def _create_resource_group(self):
        # Check if Azure CLI is installed
        version_details = AzureExecutor.get_version()
        logger.info_green(f"Your Azure CLI version: {version_details['azure-cli']}")

        # Set subscription id
        AzureExecutor.set_subscription(subscription=self.subscription)
        logger.info_green(f"Set subscription to: {self.subscription}")

        # Check and create resource group
        resource_group_details = AzureExecutor.get_resource_group(resource_group=self.resource_group)
        if resource_group_details is not None:
            logger.warning_yellow(f"Azure resource group {self.resource_group} already exists")
        else:
            AzureExecutor.create_resource_group(
                resource_group=self.resource_group,
                location=self.location
            )
            logger.info_green(f"Resource group: {self.resource_group} is created")

    def _create_vnet(self):
        logger.info("Creating vnet")

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
            resource_group=self.resource_group,
            deployment_name="vnet",
            template_file_path=abs_template_file_path,
            parameters_file_path=abs_parameters_file_path
        )

        logger.info_green("Vnet is created")

    def _build_node_image(self):
        logger.info("Building MARO Node image")

        # Build params
        resource_name = "build-node-image"
        image_name = f"{self.cluster_id}-node-image"
        vm_name = f"{self.cluster_id}-{resource_name}-vm"

        # Create ARM parameters and start deployment.
        # For simplicity, we use master_node_size as the size of build_node_image_vm here
        template_file_path = f"{GlobalPaths.ABS_MARO_GRASS_LIB}/azure/create_build_node_image_vm/template.json"
        parameters_file_path = (
            f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/azure/create_build_node_image_vm/parameters.json"
        )
        ArmTemplateParameterBuilder.create_build_node_image_vm(
            cluster_details=self.cluster_details,
            node_size=self.cluster_details["master"]["node_size"],
            export_path=parameters_file_path
        )
        AzureExecutor.start_deployment(
            resource_group=self.resource_group,
            deployment_name=resource_name,
            template_file_path=template_file_path,
            parameters_file_path=parameters_file_path
        )
        # Gracefully wait
        time.sleep(10)

        # Get public ip address
        ip_addresses = AzureExecutor.list_ip_addresses(
            resource_group=self.resource_group,
            vm_name=vm_name
        )
        public_ip_address = ip_addresses[0]["virtualMachine"]["network"]["publicIpAddresses"][0]["ipAddress"]

        # Make sure build_node_image_vm is able to connect
        self.retry_connection_and_set_ssh_port(node_ip_address=public_ip_address)

        # Run init image script
        self._sync_mkdir(path=GlobalPaths.MARO_LOCAL_TMP, node_ip_address=public_ip_address)
        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_GRASS_LIB}/scripts/build_node_image_vm/init_build_node_image_vm.py",
            remote_dir="~/",
            admin_username=self.admin_username,
            node_ip_address=public_ip_address,
            ssh_port=self.ssh_port
        )
        self.remote_init_build_node_image_vm(vm_ip_address=public_ip_address)

        # Extract image
        AzureExecutor.deallocate_vm(resource_group=self.resource_group, vm_name=vm_name)
        AzureExecutor.generalize_vm(resource_group=self.resource_group, vm_name=vm_name)
        AzureExecutor.create_image_from_vm(resource_group=self.resource_group, image_name=image_name, vm_name=vm_name)

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
        vm_name = f"{self.cluster_id}-master-vm"

        # Create ARM parameters and start deployment
        template_file_path = f"{GlobalPaths.ABS_MARO_GRASS_LIB}/azure/create_master/template.json"
        parameters_file_path = (
            f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/azure/create_master/parameters.json"
        )
        ArmTemplateParameterBuilder.create_master(
            cluster_details=self.cluster_details,
            node_size=self.cluster_details["master"]["node_size"],
            export_path=parameters_file_path
        )
        AzureExecutor.start_deployment(
            resource_group=self.resource_group,
            deployment_name="master",
            template_file_path=template_file_path,
            parameters_file_path=parameters_file_path
        )

        # Get master IP addresses
        ip_addresses = AzureExecutor.list_ip_addresses(
            resource_group=self.resource_group,
            vm_name=vm_name
        )
        public_ip_address = ip_addresses[0]["virtualMachine"]["network"]["publicIpAddresses"][0]["ipAddress"]
        private_ip_address = ip_addresses[0]["virtualMachine"]["network"]["privateIpAddresses"][0]
        hostname = vm_name
        master_details["public_ip_address"] = public_ip_address
        master_details["private_ip_address"] = private_ip_address
        master_details["hostname"] = hostname
        master_details["resource_name"] = vm_name
        self.master_public_ip_address = public_ip_address
        logger.info_green(f"You can login to your master node with: {self.admin_username}@{public_ip_address}")

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

        # Make sure master is able to connect
        self.retry_connection_and_set_ssh_port(node_ip_address=self.master_public_ip_address)

        # Create folders
        sync_paths = [
            GlobalPaths.MARO_GRASS_LIB,
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}",
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/data",
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/images",
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs",
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/schedules",
            GlobalPaths.MARO_LOCAL_TMP
        ]
        for sync_path in sync_paths:
            self._sync_mkdir(path=sync_path, node_ip_address=self.master_public_ip_address)
        self._sync_mkdir(path=GlobalPaths.MARO_GRASS_LIB, node_ip_address=self.master_public_ip_address)

        # Copy required files
        copy_files_to_node(
            local_path=GlobalPaths.MARO_GRASS_LIB,
            remote_dir=GlobalPaths.MARO_LIB,
            admin_username=self.admin_username,
            node_ip_address=self.master_public_ip_address,
            ssh_port=self.ssh_port
        )
        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}",
            remote_dir=GlobalPaths.MARO_CLUSTERS,
            admin_username=self.admin_username,
            node_ip_address=self.master_public_ip_address,
            ssh_port=self.ssh_port
        )

        # Get public key
        public_key = self.remote_get_public_key(node_ip_address=self.master_public_ip_address)

        # Remote init master
        self.remote_init_master()

        # Load master agent service
        self.remote_load_master_agent_service()

        # Save details
        master_details["public_key"] = public_key
        master_details["image_files"] = {}
        save_cluster_details(
            cluster_name=self.cluster_name,
            cluster_details=self.cluster_details
        )
        self.remote_create_master_details(master_details=master_details)

        logger.info_green("Master VM is initialized")

    # maro grass delete

    def delete(self):
        logger.info(f"Deleting cluster {self.cluster_name}")

        # Get resource list
        resource_list = AzureExecutor.list_resources(resource_group=self.resource_group)

        # Filter resources
        deletable_ids = []
        for resource_info in resource_list:
            if resource_info["name"].startswith(self.cluster_id):
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
        nodes_details = self.remote_get_nodes_details()

        # Init node_size_to_count
        node_size_to_count = collections.defaultdict(lambda: 0)
        for _, node_details in nodes_details.items():
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
                node_size=node_size
            )
        else:
            logger.warning_yellow("Replica is match, no create or delete")

    def _create_nodes(self, num: int, node_size: str) -> None:
        logger.info(f"Scaling up {num}")

        # Parallel create
        with ThreadPool(GlobalParams.PARALLELS) as pool:
            pool.starmap(
                self._create_node,
                [[node_size]] * num
            )

    def _create_node(self, node_size: str):
        # Generate node name
        node_name = generate_node_name()
        logger.info(message=f"Creating node {node_name}")

        # Create node
        self._create_vm(
            node_name=node_name,
            node_size=node_size
        )

        # Init node
        self._init_node(
            node_name=node_name
        )

        logger.info_green(message=f"Node {node_name} is created")

    def _delete_nodes(self, num: int, node_size: str) -> None:
        # Load details
        nodes_details = self.remote_get_nodes_details()

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
                "Unable to scale down.\n"
                f"Only {len(deletable_nodes)} nodes are deletable, but need to delete {num} to meet the replica"
            )

    def _create_vm(self, node_name: str, node_size: str):
        logger.info(message=f"Creating VM {node_name}")

        # Build params
        image_name = f"{self.cluster_id}-node-image"
        image_resource_id = AzureExecutor.get_image_resource_id(
            resource_group=self.resource_group,
            image_name=image_name
        )

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
            resource_group=self.resource_group,
            deployment_name=node_name,
            template_file_path=template_file_path,
            parameters_file_path=parameters_file_path
        )

        # Get node IP addresses
        ip_addresses = AzureExecutor.list_ip_addresses(
            resource_group=self.resource_group,
            vm_name=f"{self.cluster_id}-{node_name}-vm"
        )

        # Save details
        node_details = {
            "name": node_name,
            "id": node_name,
            "public_ip_address": ip_addresses[0]["virtualMachine"]["network"]["publicIpAddresses"][0]["ipAddress"],
            "private_ip_address": ip_addresses[0]["virtualMachine"]["network"]["privateIpAddresses"][0],
            "node_size": node_size,
            "resource_name": f"{self.cluster_id}-{node_name}-vm",
            "hostname": f"{self.cluster_id}-{node_name}-vm",
            "containers": {}
        }
        self.remote_create_node_details(
            node_name=node_name,
            node_details=node_details,
        )

        logger.info_green(f"VM {node_name} is created")

    def _delete_node(self, node_name: str):
        logger.info(f"Deleting node {node_name}")

        # Delete resources
        self._delete_resources(resource_name=node_name)

        # Delete azure deployment
        AzureExecutor.delete_deployment(
            resource_group=self.resource_group,
            deployment_name=node_name
        )

        # Delete parameters_file
        shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/azure/create_{node_name}")

        # Update node status
        self.remote_update_node_status(
            node_name=node_name,
            action="delete"
        )

        logger.info_green(f"Node {node_name} is deleted")

    def _init_node(self, node_name: str):
        logger.info(f"Initiating node {node_name}")

        # Load details
        node_details = self.remote_get_node_details(node_name=node_name)
        node_public_ip_address = node_details["public_ip_address"]

        # Make sure the node is able to connect
        self.retry_connection_and_set_ssh_port(node_ip_address=node_public_ip_address)

        # Copy required files
        self._sync_mkdir(path=f"{GlobalPaths.MARO_LOCAL_TMP}", node_ip_address=node_public_ip_address)
        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_GRASS_LIB}/scripts/node/init_node.py",
            remote_dir="~/",
            admin_username=self.admin_username,
            node_ip_address=node_public_ip_address,
            ssh_port=self.ssh_port
        )
        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/details.yml",
            remote_dir="~/",
            admin_username=self.admin_username,
            node_ip_address=node_public_ip_address,
            ssh_port=self.ssh_port
        )

        # Remote init node
        self.remote_init_node(
            node_name=node_name,
            node_ip_address=node_public_ip_address
        )

        # Get public key
        public_key = self.remote_get_public_key(node_ip_address=node_public_ip_address)

        # Save details
        node_details["public_key"] = public_key
        self.remote_create_node_details(
            node_name=node_name,
            node_details=node_details
        )

        # Update node status
        self.remote_update_node_status(
            node_name=node_name,
            action="create"
        )

        # Load node agent service
        self.remote_load_node_agent_service(
            node_name=node_name,
            node_ip_address=node_public_ip_address
        )

        logger.info_green(f"Node {node_name} is initialized")

    def start_node(self, replicas: int, node_size: str):
        # Get nodes details
        nodes_details = self.remote_get_nodes_details()

        # Get startable nodes
        startable_nodes = []
        for node_name, node_details in nodes_details.items():
            if node_details["node_size"] == node_size and node_details["state"]["status"] == NodeStatus.STOPPED:
                startable_nodes.append(node_name)

        # Check replicas
        if len(startable_nodes) < replicas:
            raise BadRequestError(
                f"No enough '{node_size}' nodes can be started, only {len(startable_nodes)} is able to start."
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
        node_details = self.remote_get_node_details(node_name=node_name)
        node_public_ip_address = node_details["public_ip_address"]

        # Start node
        AzureExecutor.start_vm(
            resource_group=self.resource_group,
            vm_name=f"{self.cluster_id}-{node_name}-vm"
        )

        # Update node status
        self.remote_update_node_status(
            node_name=node_name,
            action="start"
        )

        # Make sure the node is able to connect
        self.retry_connection_and_set_ssh_port(
            node_ip_address=node_public_ip_address
        )

        # Load node agent service
        self.remote_load_node_agent_service(
            node_name=node_name,
            node_ip_address=node_public_ip_address
        )

        logger.info_green(f"Node {node_name} is started")

    def stop_node(self, replicas: int, node_size: str):
        # Get nodes details
        nodes_details = self.remote_get_nodes_details()

        # Get stoppable nodes
        stoppable_nodes = []
        for node_name, node_details in nodes_details.items():
            if (
                node_details["node_size"] == node_size and
                node_details["state"]["status"] == NodeStatus.RUNNING and
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

        # Stop node
        AzureExecutor.stop_vm(
            resource_group=self.resource_group,
            vm_name=f"{self.cluster_id}-{node_name}-vm"
        )

        # Update node status
        self.remote_update_node_status(
            node_name=node_name,
            action="stop"
        )

        logger.info_green(f"Node {node_name} is stopped")

    def _get_node_size_to_spec(self) -> dict:
        # List available sizes for VMs
        specs = AzureExecutor.list_vm_sizes(location=self.location)

        # Get node_size_to_spec
        node_size_to_spec = {}
        for spec in specs:
            node_size_to_spec[spec["name"]] = spec

        return node_size_to_spec

    @staticmethod
    def _count_running_containers(node_details: dict):
        # Extract details
        containers_details = node_details["containers"]

        # Do counting
        count = 0
        for container_details in containers_details:
            if container_details["Status"] == ContainerStatus.RUNNING:
                count += 1

        return count

    # maro grass clean

    def clean(self):
        # TODO add clean redis

        # Remote clean
        self.remote_clean(parallels=GlobalParams.PARALLELS)

    # Utils

    def _delete_resources(self, resource_name: str):
        # Get resource list
        resource_list = AzureExecutor.list_resources(resource_group=self.resource_group)

        # Filter resources
        deletable_ids = []
        for resource_info in resource_list:
            if resource_info["name"].startswith(f"{self.cluster_id}-{resource_name}"):
                deletable_ids.append(resource_info["id"])

        # Delete resources
        if len(deletable_ids) > 0:
            AzureExecutor.delete_resources(resources=deletable_ids)


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
            parameters["sshDestinationPorts"]["value"] = (
                [f"{ssh_port}"] if ssh_port == GlobalParams.DEFAULT_SSH_PORT
                else [GlobalParams.DEFAULT_SSH_PORT, f"{ssh_port}"]
            )

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
            parameters["sshDestinationPorts"]["value"] = (
                [f"{ssh_port}"] if ssh_port == GlobalParams.DEFAULT_SSH_PORT
                else [GlobalParams.DEFAULT_SSH_PORT, f"{ssh_port}"]
            )

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
            parameters["sshDestinationPorts"]["value"] = (
                [f"{ssh_port}"] if ssh_port == GlobalParams.DEFAULT_SSH_PORT
                else [GlobalParams.DEFAULT_SSH_PORT, f"{ssh_port}"]
            )

        # Export parameters if the path is set
        if export_path:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w") as fw:
                json.dump(base_parameters, fw, indent=4)

        return base_parameters
