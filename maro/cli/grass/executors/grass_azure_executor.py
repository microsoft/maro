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
from maro.cli.grass.utils.file_synchronizer import FileSynchronizer
from maro.cli.grass.utils.master_api_client import MasterApiClientV1
from maro.cli.grass.utils.params import ContainerStatus, GrassParams, NodeStatus, GrassPaths
from maro.cli.utils.azure_controller import AzureController
from maro.cli.utils.deployment_validator import DeploymentValidator
from maro.cli.utils.details_reader import DetailsReader
from maro.cli.utils.details_writer import DetailsWriter
from maro.cli.utils.name_creator import NameCreator
from maro.cli.utils.params import GlobalParams, GlobalPaths
from maro.utils.exception.cli_exception import BadRequestError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class GrassAzureExecutor(GrassExecutor):

    def __init__(self, cluster_name: str):
        super().__init__(cluster_details=DetailsReader.load_cluster_details(cluster_name=cluster_name))

        # Cloud configs
        self.subscription = self.cluster_details["cloud"]["subscription"]
        self.resource_group = self.cluster_details["cloud"]["resource_group"]
        self.location = self.cluster_details["cloud"]["location"]
        self.default_username = self.cluster_details["cloud"]["default_username"]

        # Connection configs
        self.ssh_port = self.cluster_details["connection"]["ssh"]["port"]
        self.api_server_port = self.cluster_details["connection"]["api_server"]["port"]

    # maro grass create

    @staticmethod
    def create(create_deployment: dict):
        logger.info("Creating cluster")

        # Get standardized cluster_details
        cluster_details = GrassAzureExecutor._standardize_cluster_details(create_deployment=create_deployment)
        cluster_name = cluster_details["name"]
        if os.path.isdir(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}"):
            raise BadRequestError(f"Cluster '{cluster_name}' is exist")

        # Start creating
        try:
            GrassAzureExecutor._create_resource_group(cluster_details=cluster_details)
            GrassAzureExecutor._create_vnet(cluster_details=cluster_details)

            # Simultaneously capture image and init master
            build_node_image_thread = threading.Thread(
                target=GrassAzureExecutor._build_node_image,
                args=(cluster_details,)
            )
            build_node_image_thread.start()
            create_and_init_master_thread = threading.Thread(
                target=GrassAzureExecutor._create_and_init_master,
                args=(cluster_details,)
            )
            create_and_init_master_thread.start()
            build_node_image_thread.join()
            create_and_init_master_thread.join()

            # local save cluster after initialization
            DetailsWriter.save_cluster_details(cluster_name=cluster_name, cluster_details=cluster_details)
        except Exception as e:
            # If failed, remove details folder, then raise
            shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}")
            logger.error_red(f"Failed to create cluster '{cluster_name}'")
            raise e

        logger.info_green(f"Cluster '{cluster_name}' is created")

    @staticmethod
    def _standardize_cluster_details(create_deployment: dict) -> dict:
        samba_password = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20))
        optional_key_to_value = {
            "root['master']['redis']": {"port": GlobalParams.DEFAULT_REDIS_PORT},
            "root['master']['redis']['port']": GlobalParams.DEFAULT_REDIS_PORT,
            "root['master']['fluentd']": {"port": GlobalParams.DEFAULT_FLUENTD_PORT},
            "root['master']['fluentd']['port']": GlobalParams.DEFAULT_FLUENTD_PORT,
            "root['master']['samba']": {"password": samba_password},
            "root['master']['samba']['password']": samba_password,
            "root['connection']": {
                "ssh": {"port": GlobalParams.DEFAULT_SSH_PORT},
                "api_server": {"port": GrassParams.DEFAULT_API_SERVER_PORT},
            },
            "root['connection']['ssh']": {"port": GlobalParams.DEFAULT_SSH_PORT},
            "root['connection']['ssh']['port']": GlobalParams.DEFAULT_SSH_PORT,
            "root['connection']['api_server']": {"port": GrassParams.DEFAULT_API_SERVER_PORT},
            "root['connection']['api_server']['port']": GrassParams.DEFAULT_API_SERVER_PORT
        }
        with open(f"{GrassPaths.ABS_MARO_GRASS_LIB}/deployments/internal/grass_azure_create.yml") as fr:
            create_deployment_template = yaml.safe_load(fr)
        DeploymentValidator.validate_and_fill_dict(
            template_dict=create_deployment_template,
            actual_dict=create_deployment,
            optional_key_to_value=optional_key_to_value
        )

        # Init runtime fields.
        create_deployment["id"] = NameCreator.create_cluster_id()

        return create_deployment

    @staticmethod
    def _create_resource_group(cluster_details: dict) -> None:
        # Load params
        subscription = cluster_details["cloud"]["subscription"]
        resource_group = cluster_details["cloud"]["resource_group"]

        # Check if Azure CLI is installed, and print version
        version_details = AzureController.get_version()
        logger.info_green(f"Your Azure CLI version: {version_details['azure-cli']}")

        # Set subscription id
        AzureController.set_subscription(subscription=subscription)
        logger.info_green(f"Set subscription to '{subscription}'")

        # Check and create resource group
        resource_group_details = AzureController.get_resource_group(resource_group=resource_group)
        if resource_group_details is not None:
            logger.warning_yellow(f"Azure resource group '{resource_group}' already exists")
        else:
            AzureController.create_resource_group(
                resource_group=resource_group,
                location=cluster_details["cloud"]["location"]
            )
            logger.info_green(f"Resource group '{resource_group}' is created")

    @staticmethod
    def _create_vnet(cluster_details: dict) -> None:
        logger.info("Creating vnet")

        # Create ARM parameters and start deployment
        template_file_path = f"{GrassPaths.ABS_MARO_GRASS_LIB}/modes/azure/create_vnet/template.json"
        parameters_file_path = (
            f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_details['name']}/modes/azure/create_vnet/parameters.json"
        )
        ArmTemplateParameterBuilder.create_vnet(
            cluster_details=cluster_details,
            export_path=parameters_file_path
        )
        AzureController.start_deployment(
            resource_group=cluster_details["cloud"]["resource_group"],
            deployment_name="vnet",
            template_file_path=template_file_path,
            parameters_file_path=parameters_file_path
        )

        logger.info_green("Vnet is created")

    @staticmethod
    def _build_node_image(cluster_details: dict) -> None:
        logger.info("Building MARO Node image")

        # Build params
        resource_name = "build-node-image"
        image_name = f"{cluster_details['id']}-node-image"
        vm_name = f"{cluster_details['id']}-{resource_name}-vm"

        # Create ARM parameters and start deployment.
        # For simplicity, we use master_node_size as the size of build_node_image_vm here
        template_file_path = f"{GrassPaths.ABS_MARO_GRASS_LIB}/modes/azure/create_build_node_image_vm/template.json"
        parameters_file_path = (
            f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_details['name']}"
            f"/modes/azure/create_build_node_image_vm/parameters.json"
        )
        ArmTemplateParameterBuilder.create_build_node_image_vm(
            cluster_details=cluster_details,
            node_size=cluster_details["master"]["node_size"],
            export_path=parameters_file_path
        )
        AzureController.start_deployment(
            resource_group=cluster_details["cloud"]["resource_group"],
            deployment_name=resource_name,
            template_file_path=template_file_path,
            parameters_file_path=parameters_file_path
        )
        # Gracefully wait
        time.sleep(10)

        # Get public ip address
        ip_addresses = AzureController.list_ip_addresses(
            resource_group=cluster_details["cloud"]["resource_group"],
            vm_name=vm_name
        )
        public_ip_address = ip_addresses[0]["virtualMachine"]["network"]["publicIpAddresses"][0]["ipAddress"]

        # Make sure build_node_image_vm is able to connect
        GrassAzureExecutor.retry_connection(
            node_username=cluster_details["cloud"]["default_username"],
            node_hostname=public_ip_address,
            node_ssh_port=cluster_details["connection"]["ssh"]["port"]
        )

        # Run init image script
        FileSynchronizer.copy_files_to_node(
            local_path=f"{GrassPaths.MARO_GRASS_LIB}/scripts/build_node_image_vm/init_build_node_image_vm.py",
            remote_dir="~/",
            node_username=cluster_details["cloud"]["default_username"],
            node_hostname=public_ip_address,
            node_ssh_port=cluster_details["connection"]["ssh"]["port"]
        )
        GrassAzureExecutor.remote_init_build_node_image_vm(
            node_username=cluster_details["cloud"]["default_username"],
            node_hostname=public_ip_address,
            node_ssh_port=cluster_details["connection"]["ssh"]["port"]
        )

        # Extract image
        AzureController.deallocate_vm(resource_group=cluster_details["cloud"]["resource_group"], vm_name=vm_name)
        AzureController.generalize_vm(resource_group=cluster_details["cloud"]["resource_group"], vm_name=vm_name)
        AzureController.create_image_from_vm(
            resource_group=cluster_details["cloud"]["resource_group"],
            image_name=image_name,
            vm_name=vm_name
        )

        # Delete resources
        GrassAzureExecutor._delete_resources(
            resource_group=cluster_details["cloud"]["resource_group"],
            resource_name=resource_name,
            cluster_id=cluster_details["id"]
        )

        logger.info_green("MARO Node Image is built")

    @staticmethod
    def _create_and_init_master(cluster_details: dict) -> None:
        logger.info("Creating MARO Master")

        GrassAzureExecutor._create_master_vm(cluster_details=cluster_details)
        GrassAzureExecutor._init_master(cluster_details=cluster_details)

        # Remote create master after initialization
        MasterApiClientV1(
            master_hostname=cluster_details["master"]["public_ip_address"],
            master_api_server_port=cluster_details["master"]["api_server"]["port"]
        ).create_master(master_details=cluster_details["master"])

        logger.info_green("MARO Master is created")

    @staticmethod
    def _create_master_vm(cluster_details: dict) -> None:
        logger.info("Creating Master VM")

        # Build params
        vm_name = f"{cluster_details['id']}-master-vm"

        # Create ARM parameters and start deployment
        template_file_path = f"{GrassPaths.ABS_MARO_GRASS_LIB}/modes/azure/create_master/template.json"
        parameters_file_path = (
            f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_details['name']}/modes/azure/create_master/parameters.json"
        )
        ArmTemplateParameterBuilder.create_master(
            cluster_details=cluster_details,
            node_size=cluster_details["master"]["node_size"],
            export_path=parameters_file_path
        )
        AzureController.start_deployment(
            resource_group=cluster_details["cloud"]["resource_group"],
            deployment_name="master",
            template_file_path=template_file_path,
            parameters_file_path=parameters_file_path
        )

        # Get master IP addresses
        ip_addresses = AzureController.list_ip_addresses(
            resource_group=cluster_details["cloud"]["resource_group"],
            vm_name=vm_name
        )
        public_ip_address = ip_addresses[0]["virtualMachine"]["network"]["publicIpAddresses"][0]["ipAddress"]
        private_ip_address = ip_addresses[0]["virtualMachine"]["network"]["privateIpAddresses"][0]

        # Get other params and fill them to master_details
        hostname = vm_name
        username = cluster_details["cloud"]["default_username"]
        cluster_details["master"]["hostname"] = hostname
        cluster_details["master"]["username"] = username
        cluster_details["master"]["public_ip_address"] = public_ip_address
        cluster_details["master"]["private_ip_address"] = private_ip_address
        cluster_details["master"]["resource_name"] = vm_name
        cluster_details["master"]["ssh"] = {"port": cluster_details["connection"]["ssh"]["port"]}
        cluster_details["master"]["api_server"] = {"port": cluster_details["connection"]["api_server"]["port"]}
        logger.info_green(f"You can login to your master node with: {username}@{public_ip_address}")

        logger.info_green("Master VM is created")

    @staticmethod
    def _init_master(cluster_details: dict) -> None:
        logger.info("Initializing Master VM")

        # Make sure master is able to connect
        GrassAzureExecutor.retry_connection(
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
            GrassPaths.MARO_GRASS_LIB: f"{GlobalPaths.MARO_SHARED}/lib",
            f"{GlobalPaths.MARO_CLUSTERS}/{cluster_details['name']}": f"{GlobalPaths.MARO_SHARED}/clusters"
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
        GrassAzureExecutor.remote_init_master(
            master_username=cluster_details["master"]["username"],
            master_hostname=cluster_details["master"]["public_ip_address"],
            master_ssh_port=cluster_details["master"]["ssh"]["port"],
            cluster_name=cluster_details["name"]
        )
        # Gracefully wait
        time.sleep(10)

        logger.info_green("Master VM is initialized")

    # maro grass delete

    def delete(self):
        logger.info(f"Deleting cluster '{self.cluster_name}'")

        # Get resource list
        resource_list = AzureController.list_resources(resource_group=self.resource_group)

        # Filter resources
        deletable_ids = []
        for resource_info in resource_list:
            if resource_info["name"].startswith(self.cluster_id):
                deletable_ids.append(resource_info["id"])

        # Delete resources
        if len(deletable_ids) > 0:
            AzureController.delete_resources(resources=deletable_ids)

        # Delete cluster folder
        shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}")

        logger.info_green(f"Cluster '{self.cluster_name}' is deleted")

    # maro grass node

    def scale_node(self, replicas: int, node_size: str):
        # Load details
        nodes_details = self.master_api_client.list_nodes()

        # Init node_size_to_count
        node_size_to_count = collections.defaultdict(lambda: 0)
        for node_details in nodes_details:
            node_size_to_count[node_details["node_size"]] += 1

        # Get node_size_to_spec
        node_size_to_spec = self._get_node_size_to_spec()
        if node_size not in node_size_to_spec:
            raise BadRequestError(f"Invalid node_size '{node_size}'")

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
        node_name = NameCreator.create_node_name()
        logger.info(message=f"Creating node '{node_name}'")

        # Create node
        join_cluster_deployment = self._create_vm(
            node_name=node_name,
            node_size=node_size
        )

        # Start joining cluster
        self._join_cluster(node_details=join_cluster_deployment["node"])

        logger.info_green(message=f"Node '{node_name}' is created")

    def _delete_nodes(self, num: int, node_size: str) -> None:
        # Load details
        nodes_details = self.master_api_client.list_nodes()

        # Get deletable_nodes and check, TODO: consider to add -f
        deletable_nodes = []
        for node_details in nodes_details:
            if node_details["node_size"] == node_size and len(node_details["containers"]) == 0:
                deletable_nodes.append(node_details["name"])
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

    def _create_vm(self, node_name: str, node_size: str) -> dict:
        logger.info(message=f"Creating VM '{node_name}'")

        # Build params
        image_name = f"{self.cluster_id}-node-image"
        image_resource_id = AzureController.get_image_resource_id(
            resource_group=self.resource_group,
            image_name=image_name
        )

        # Create ARM parameters and start deployment
        template_file_path = f"{GrassPaths.ABS_MARO_GRASS_LIB}/modes/azure/create_node/template.json"
        parameters_file_path = (
            f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/modes/azure/create_{node_name}/parameters.json"
        )
        ArmTemplateParameterBuilder.create_node(
            node_name=node_name,
            cluster_details=self.cluster_details,
            node_size=node_size,
            image_resource_id=image_resource_id,
            export_path=parameters_file_path
        )
        AzureController.start_deployment(
            resource_group=self.resource_group,
            deployment_name=node_name,
            template_file_path=template_file_path,
            parameters_file_path=parameters_file_path
        )

        # Get node IP addresses
        ip_addresses = AzureController.list_ip_addresses(
            resource_group=self.resource_group,
            vm_name=f"{self.cluster_id}-{node_name}-vm"
        )

        logger.info_green(f"VM '{node_name}' is created")

        # Build join_cluster_deployment.
        join_cluster_deployment = {
            "mode": "grass/azure",
            "master": {
                "hostname": self.master_hostname,
                "api_server": {
                    "port": self.api_server_port
                }
            },
            "node": {
                "name": node_name,
                "id": node_name,
                "username": self.default_username,
                "public_ip_address": ip_addresses[0]["virtualMachine"]["network"]["publicIpAddresses"][0]["ipAddress"],
                "private_ip_address": ip_addresses[0]["virtualMachine"]["network"]["privateIpAddresses"][0],
                "node_size": node_size,
                "resource_name": f"{self.cluster_id}-{node_name}-vm",
                "hostname": f"{self.cluster_id}-{node_name}-vm",
                "resources": {
                    "cpu": "all",
                    "memory": "all",
                    "gpu": "all"
                },
                "api_server": {
                    "port": self.api_server_port
                },
                "ssh": {
                    "port": self.ssh_port
                }
            }
        }
        with open(f"{GlobalPaths.ABS_MARO_LOCAL_TMP}/join_{node_name}.yml", "w") as fw:
            yaml.safe_dump(data=join_cluster_deployment, stream=fw)

        return join_cluster_deployment

    def _delete_node(self, node_name: str):
        logger.info(f"Deleting node '{node_name}'")

        # Delete node
        self.master_api_client.delete_node(node_name=node_name)

        # Delete resources
        self._delete_resources(
            resource_group=self.resource_group,
            cluster_id=self.cluster_id,
            resource_name=node_name
        )

        # Delete azure deployment
        AzureController.delete_deployment(
            resource_group=self.resource_group,
            deployment_name=node_name
        )

        # Delete parameters_file
        shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/modes/azure/create_{node_name}")

        logger.info_green(f"Node '{node_name}' is deleted")

    def _join_cluster(self, node_details: dict):
        node_name = node_details["name"]

        logger.info(f"Node '{node_name}' is joining the cluster '{self.cluster_name}'")

        # Make sure the node is able to connect
        self.retry_connection(
            node_username=node_details["username"],
            node_hostname=node_details["public_ip_address"],
            node_ssh_port=node_details["ssh"]["port"]
        )

        # Copy required files
        local_path_to_remote_dir = {f"{GlobalPaths.MARO_LOCAL_TMP}/join_{node_name}.yml": "~/"}
        for local_path, remote_dir in local_path_to_remote_dir.items():
            FileSynchronizer.copy_files_to_node(
                local_path=local_path,
                remote_dir=remote_dir,
                node_username=node_details["username"],
                node_hostname=node_details["public_ip_address"],
                node_ssh_port=node_details["ssh"]["port"]
            )

        #  Remote join cluster
        self.remote_join_cluster(
            node_username=node_details["username"],
            node_hostname=node_details["public_ip_address"],
            node_ssh_port=node_details["ssh"]["port"],
            master_hostname=self.master_public_ip_address,
            master_api_server_port=self.master_api_server_port,
            deployment_path=f"~/join_{node_name}.yml"
        )

        # Delete local join_deployment.
        os.remove(f"{GlobalPaths.ABS_MARO_LOCAL_TMP}/join_{node_name}.yml")

        logger.info_green(f"Node '{node_name}' is joined")

    def start_node(self, replicas: int, node_size: str):
        # Get nodes details
        nodes_details = self.master_api_client.list_nodes()

        # Get startable nodes
        startable_nodes = []
        for node_details in nodes_details:
            if node_details["node_size"] == node_size and node_details["state"]["status"] == NodeStatus.STOPPED:
                startable_nodes.append(node_details["name"])

        # Check replicas
        if len(startable_nodes) < replicas:
            raise BadRequestError(
                f"No enough '{node_size}' nodes can be started, only {len(startable_nodes)} is able to start"
            )

        # Parallel start
        params = [[startable_node] for startable_node in startable_nodes[:replicas]]
        with ThreadPool(GlobalParams.PARALLELS) as pool:
            pool.starmap(
                self._start_node,
                params
            )

    def _start_node(self, node_name: str):
        logger.info(f"Starting node '{node_name}'")

        # Start node vm
        AzureController.start_vm(
            resource_group=self.resource_group,
            vm_name=f"{self.cluster_id}-{node_name}-vm"
        )

        # Start node
        self.master_api_client.start_node(node_name=node_name)

        logger.info_green(f"Node '{node_name}' is started")

    def stop_node(self, replicas: int, node_size: str):
        # Get nodes details
        nodes_details = self.master_api_client.list_nodes()

        # Get stoppable nodes
        stoppable_nodes_details = []
        for node_details in nodes_details:
            if (
                node_details["node_size"] == node_size and
                node_details["state"]["status"] == NodeStatus.RUNNING and
                self._count_running_containers(node_details) == 0
            ):
                stoppable_nodes_details.append(node_details)

        # Check replicas
        if len(stoppable_nodes_details) < replicas:
            raise BadRequestError(
                f"No more '{node_size}' nodes can be stopped, only {len(stoppable_nodes_details)} are stoppable"
            )

        # Parallel stop
        params = [[node_details] for node_details in stoppable_nodes_details[:replicas]]
        with ThreadPool(GlobalParams.PARALLELS) as pool:
            pool.starmap(
                self._stop_node,
                params
            )

    def _stop_node(self, node_details: dict):
        node_name = node_details["name"]

        logger.info(f"Stopping node '{node_name}'")

        # Stop node
        self.master_api_client.stop_node(node_name=node_name)

        # Stop node vm
        AzureController.stop_vm(
            resource_group=self.resource_group,
            vm_name=f"{self.cluster_id}-{node_name}-vm"
        )

        logger.info_green(f"Node '{node_name}' is stopped")

    def _get_node_size_to_spec(self) -> dict:
        # List available sizes for VMs
        specs = AzureController.list_vm_sizes(location=self.location)

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
        # Remote clean jobs
        self.master_api_client.clean_jobs()

    # Utils

    @staticmethod
    def _delete_resources(resource_group: str, cluster_id: int, resource_name: str):
        # Get resource list
        resource_list = AzureController.list_resources(resource_group=resource_group)

        # Filter resources
        deletable_ids = []
        for resource_info in resource_list:
            if resource_info["name"].startswith(f"{cluster_id}-{resource_name}"):
                deletable_ids.append(resource_info["id"])

        # Delete resources
        if len(deletable_ids) > 0:
            AzureController.delete_resources(resources=deletable_ids)


class ArmTemplateParameterBuilder:
    @staticmethod
    def create_vnet(cluster_details: dict, export_path: str) -> dict:
        # Get params
        cluster_id = cluster_details["id"]
        location = cluster_details["cloud"]["location"]

        # Load and update parameters
        with open(f"{GrassPaths.ABS_MARO_GRASS_LIB}/modes/azure/create_vnet/parameters.json", "r") as f:
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
        default_username = cluster_details["cloud"]["default_username"]
        default_public_key = cluster_details["cloud"]["default_public_key"]
        ssh_port = cluster_details["connection"]["ssh"]["port"]
        api_server_port = cluster_details["connection"]["api_server"]["port"]

        # Load and update parameters
        with open(f"{GrassPaths.ABS_MARO_GRASS_LIB}/modes/azure/create_master/parameters.json", "r") as f:
            base_parameters = json.load(f)
            parameters = base_parameters["parameters"]
            parameters["location"]["value"] = location
            parameters["networkInterfaceName"]["value"] = f"{cluster_id}-{resource_name}-nic"
            parameters["networkSecurityGroupName"]["value"] = f"{cluster_id}-{resource_name}-nsg"
            parameters["virtualNetworkName"]["value"] = f"{cluster_id}-vnet"
            parameters["publicIpAddressName"]["value"] = f"{cluster_id}-{resource_name}-pip"
            parameters["virtualMachineName"]["value"] = f"{cluster_id}-{resource_name}-vm"
            parameters["virtualMachineSize"]["value"] = node_size
            parameters["adminUsername"]["value"] = default_username
            parameters["adminPublicKey"]["value"] = default_public_key
            parameters["sshDestinationPorts"]["value"] = (
                [f"{ssh_port}"] if ssh_port == GlobalParams.DEFAULT_SSH_PORT
                else [GlobalParams.DEFAULT_SSH_PORT, f"{ssh_port}"]
            )
            parameters["masterApiServerDestinationPorts"]["value"] = [f"{api_server_port}"]

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
        default_username = cluster_details["cloud"]["default_username"]
        default_public_key = cluster_details["cloud"]["default_public_key"]
        ssh_port = cluster_details["connection"]["ssh"]["port"]

        # Load and update parameters
        with open(
            f"{GrassPaths.ABS_MARO_GRASS_LIB}/modes/azure/create_build_node_image_vm/parameters.json", "r"
        ) as f:
            base_parameters = json.load(f)
            parameters = base_parameters["parameters"]
            parameters["location"]["value"] = location
            parameters["networkInterfaceName"]["value"] = f"{cluster_id}-{resource_name}-nic"
            parameters["networkSecurityGroupName"]["value"] = f"{cluster_id}-{resource_name}-nsg"
            parameters["virtualNetworkName"]["value"] = f"{cluster_id}-vnet"
            parameters["publicIpAddressName"]["value"] = f"{cluster_id}-{resource_name}-pip"
            parameters["virtualMachineName"]["value"] = f"{cluster_id}-{resource_name}-vm"
            parameters["virtualMachineSize"]["value"] = node_size
            parameters["adminUsername"]["value"] = default_username
            parameters["adminPublicKey"]["value"] = default_public_key
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
        default_username = cluster_details["cloud"]["default_username"]
        default_public_key = cluster_details["cloud"]["default_public_key"]
        ssh_port = cluster_details["connection"]["ssh"]["port"]

        # Load and update parameters
        with open(f"{GrassPaths.ABS_MARO_GRASS_LIB}/modes/azure/create_node/parameters.json", "r") as f:
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
            parameters["adminUsername"]["value"] = default_username
            parameters["adminPublicKey"]["value"] = default_public_key
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
