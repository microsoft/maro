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
        cluster_name = create_deployment['name']
        if os.path.isdir(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}")):
            raise CliException(f"Cluster {cluster_name} is exist")
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
                f'{GlobalPaths.MARO_GRASS_LIB}/deployments/internal/grass-azure-create.yml')) as fr:
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
            self._create_master()
            self._init_master()
        except Exception as e:
            # If failed, remove details folder, then raise
            rmtree(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}"))
            raise e

        logger.info_green(f"Cluster {self.cluster_name} is created")

    def _set_cluster_id(self):
        # Load details
        cluster_details = self.cluster_details

        # Set cluster id
        cluster_details['id'] = generate_cluster_id()

        # Save details
        save_cluster_details(
            cluster_name=self.cluster_name,
            cluster_details=cluster_details
        )

    def _create_resource_group(self):
        # Load and reload details
        cluster_details = self.cluster_details
        subscription = cluster_details['cloud']['subscription']
        resource_group = cluster_details['cloud']['resource_group']
        location = cluster_details['cloud']['location']

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

    def _create_master(self):
        logger.info("Creating master VM")

        # Load details
        cluster_details = self.cluster_details
        master_details = cluster_details['master']
        cluster_id = cluster_details['id']
        resource_group = cluster_details['cloud']['resource_group']
        admin_username = cluster_details['user']['admin_username']
        node_size = cluster_details['master']['node_size']

        # Create ARM parameters
        self._create_deployment_parameters(
            node_name='master',
            cluster_details=cluster_details,
            node_size=node_size,
            export_dir=os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/parameters")
        )

        # Start deployment
        template_file_location = f"{GlobalPaths.MARO_GRASS_LIB}/azure/grass-create-default-node-template.json"
        parameters_file_location = f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/parameters/master.json"
        AzureExecutor.start_deployment(
            resource_group=resource_group,
            deployment_name='master',
            template_file=template_file_location,
            parameters_file=parameters_file_location
        )

        # Get master IP addresses
        ip_addresses = AzureExecutor.list_ip_addresses(
            resource_group=resource_group,
            vm_name=f"{cluster_id}-master-vm"
        )
        public_ip_address = ip_addresses[0]["virtualMachine"]["network"]['publicIpAddresses'][0]['ipAddress']
        private_ip_address = ip_addresses[0]["virtualMachine"]["network"]['privateIpAddresses'][0]
        hostname = f"{cluster_id}-master-vm"
        master_details['public_ip_address'] = public_ip_address
        master_details['private_ip_address'] = private_ip_address
        master_details['hostname'] = hostname
        master_details['resource_name'] = f"{cluster_id}-master-vm"
        logger.info_green(f"You can login to your master node with: ssh {admin_username}@{public_ip_address}")

        # Save details
        save_cluster_details(
            cluster_name=self.cluster_name,
            cluster_details=cluster_details,
            sync=False
        )

        logger.info_green("Master VM is created")

    def _init_master(self):
        logger.info("Initializing master node")

        # Load details
        cluster_details = self.cluster_details
        master_details = cluster_details['master']
        admin_username = cluster_details['user']['admin_username']
        master_public_ip_address = cluster_details['master']['public_ip_address']

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
        master_details['image_files'] = {}
        save_cluster_details(
            cluster_name=self.cluster_name,
            cluster_details=cluster_details
        )
        self.grass_executor.remote_set_master_details(master_details=cluster_details['master'])

        logger.info_green("Master node is initialized")

    @staticmethod
    def _create_deployment_parameters(node_name: str, cluster_details: dict, node_size: str, export_dir: str):
        # Extract variables
        cluster_id = cluster_details['id']
        location = cluster_details['cloud']['location']
        admin_username = cluster_details['user']['admin_username']
        admin_public_key = cluster_details['user']['admin_public_key']

        # Mkdir
        os.makedirs(export_dir, exist_ok=True)

        # Load and update parameters
        with open(os.path.expanduser(f"{GlobalPaths.MARO_GRASS_LIB}/azure/grass-create-parameters.json"), 'r') as f:
            base_parameters = json.load(f)
        with open(export_dir + f"/{node_name}.json", 'w') as fw:
            parameters = base_parameters['parameters']
            parameters['location']['value'] = location
            parameters['networkInterfaceName']['value'] = f"{cluster_id}-{node_name}-nic"
            parameters['networkSecurityGroupName']['value'] = f"{cluster_id}-{node_name}-nsg"
            parameters['virtualNetworkName']['value'] = f"{cluster_id}-vnet"
            parameters['publicIpAddressName']['value'] = f"{cluster_id}-{node_name}-pip"
            parameters['virtualMachineName']['value'] = f"{cluster_id}-{node_name}-vm"
            parameters['virtualMachineSize']['value'] = node_size
            parameters['adminUsername']['value'] = admin_username
            parameters['adminPublicKey']['value'] = admin_public_key
            json.dump(base_parameters, fw, indent=4)

    # maro grass delete

    def delete(self):
        # Load details
        cluster_name = self.cluster_name
        cluster_details = load_cluster_details(cluster_name=cluster_name)
        cluster_id = cluster_details['id']
        resource_group = cluster_details['cloud']['resource_group']

        logger.info(f"Deleting cluster {cluster_name}")

        # Get resource list
        resource_list = AzureExecutor.list_resources(resource_group=resource_group)

        # Filter resources
        deletable_ids = []
        for resource_info in resource_list:
            if resource_info['name'].startswith(cluster_id):
                deletable_ids.append(resource_info['id'])

        # Delete resources
        if len(deletable_ids) > 0:
            AzureExecutor.delete_resources(resources=deletable_ids)

        # Delete cluster folder
        rmtree(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}"))

        logger.info_green(f"Cluster {cluster_name} is deleted")

    # maro grass node

    def scale_node(self, replicas: int, node_size: str):
        # Load details
        nodes_details = self.grass_executor.remote_get_nodes_details()

        # Init node_size_to_count
        node_size_to_count = collections.defaultdict(lambda: 0)
        for node_name, node_details in nodes_details.items():
            node_size_to_count[node_details['node_size']] += 1

        # Get node_size_to_spec
        node_size_to_spec = self._get_node_size_to_spec()
        if node_size not in node_size_to_spec:
            raise CliException(f"Invalid node_size {node_size}")

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
            if node_details['node_size'] == node_size and len(node_details['containers']) == 0:
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
                f"Unable to scale down."
                f" Only {len(deletable_nodes)} are deletable, but need to delete {num} to meet the replica"
            )

    def _create_vm(self, node_name: str, node_size: str, node_size_to_spec: dict):
        logger.info(message=f"Creating VM {node_name}")

        # Load details
        cluster_details = self.cluster_details
        location = cluster_details['cloud']['location']
        cluster_id = cluster_details['id']
        resource_group = cluster_details['cloud']['resource_group']

        # Create ARM parameters
        GrassAzureExecutor._create_deployment_parameters(
            node_name=node_name,
            cluster_details=cluster_details,
            node_size=node_size,
            export_dir=os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/parameters")
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

        # Start deployment
        if gpu_nums > 0:
            template_file_location = f"{GlobalPaths.MARO_GRASS_LIB}/azure/grass-create-gpu-node-template.json"
        else:
            template_file_location = f"{GlobalPaths.MARO_GRASS_LIB}/azure/grass-create-default-node-template.json"
        parameters_file_location = f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/parameters/{node_name}.json"
        AzureExecutor.start_deployment(
            resource_group=resource_group,
            deployment_name=node_name,
            template_file=template_file_location,
            parameters_file=parameters_file_location
        )

        # Get node IP addresses
        ip_addresses = AzureExecutor.list_ip_addresses(
            resource_group=resource_group,
            vm_name=f"{cluster_id}-{node_name}-vm"
        )

        # Save details
        node_details = {
            'public_ip_address': ip_addresses[0]["virtualMachine"]["network"]['publicIpAddresses'][0]['ipAddress'],
            'private_ip_address': ip_addresses[0]["virtualMachine"]["network"]['privateIpAddresses'][0],
            'node_size': node_size,
            'resource_name': f"{cluster_id}-{node_name}-vm",
            'hostname': f"{cluster_id}-{node_name}-vm",
            'resources': {
                'cpu': node_size_to_spec[node_size]['numberOfCores'],
                'memory': node_size_to_spec[node_size]['memoryInMb'],
                'gpu': gpu_nums
            },
            'containers': {}
        }
        self.grass_executor.remote_set_node_details(
            node_name=node_name,
            node_details=node_details,
        )

        logger.info_green(f"VM {node_name} is created")

    def _delete_node(self, node_name: str):
        logger.info(f"Deleting node {node_name}")

        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']
        resource_group = cluster_details['cloud']['resource_group']

        # Get resource list
        resource_list = AzureExecutor.list_resources(resource_group=resource_group)

        # Filter resources
        deletable_ids = []
        for resource_info in resource_list:
            if resource_info['name'].startswith(f"{cluster_id}-{node_name}"):
                deletable_ids.append(resource_info['id'])

        # Delete resources
        if len(deletable_ids) > 0:
            AzureExecutor.delete_resources(resources=deletable_ids)

        # Delete azure deployment
        AzureExecutor.delete_deployment(
            resource_group=resource_group,
            deployment_name=node_name
        )

        # Delete parameters_file
        parameters_file_location = f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/parameters/{node_name}.json"
        command = f"rm {parameters_file_location}"
        _ = SubProcess.run(command)

        # Update node status
        self.grass_executor.remote_update_node_status(
            node_name=node_name,
            action='delete'
        )

        logger.info_green(f"Node {node_name} is deleted")

    def _init_node(self, node_name: str):
        logger.info(f"Initiating node {node_name}")

        # Load details
        cluster_details = self.cluster_details
        admin_username = cluster_details['user']['admin_username']
        node_details = self.grass_executor.remote_get_node_details(node_name=node_name)
        node_public_ip_address = node_details['public_ip_address']

        # Make sure the node is able to connect
        self.grass_executor.retry_until_connected(node_ip_address=node_public_ip_address)

        # Copy required files
        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_GRASS_LIB}/scripts/init_node.py",
            remote_dir="~/",
            admin_username=admin_username, node_ip_address=node_public_ip_address)
        copy_files_to_node(
            local_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/details.yml",
            remote_dir="~/",
            admin_username=admin_username, node_ip_address=node_public_ip_address)

        # Remote init node
        self.grass_executor.remote_init_node(
            node_name=node_name,
            node_ip_address=node_public_ip_address
        )

        # Get public key
        public_key = self.grass_executor.remote_get_public_key(node_ip_address=node_public_ip_address)

        # Save details
        node_details['public_key'] = public_key
        self.grass_executor.remote_set_node_details(
            node_name=node_name,
            node_details=node_details
        )

        # Update node status
        self.grass_executor.remote_update_node_status(
            node_name=node_name,
            action='create'
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
            if node_details['node_size'] == node_size and node_details['state'] == 'Stopped':
                startable_nodes.append(node_name)

        # Check replicas
        if len(startable_nodes) < replicas:
            raise CliException(f"No enough {node_size} nodes can be started")

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
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']
        resource_group = cluster_details['cloud']['resource_group']
        node_details = self.grass_executor.remote_get_node_details(node_name=node_name)
        node_public_ip_address = node_details['public_ip_address']

        # Start node
        AzureExecutor.start_vm(
            resource_group=resource_group,
            vm_name=f"{cluster_id}-{node_name}-vm"
        )

        # Update node status
        self.grass_executor.remote_update_node_status(
            node_name=node_name,
            action='start'
        )

        # Make sure the node is able to connect
        self.grass_executor.retry_until_connected(
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
            if node_details['node_size'] == node_size and \
                    node_details['state'] == 'Running' and \
                    self._count_running_containers(node_details) == 0:
                stoppable_nodes.append(node_name)

        # Check replicas
        if len(stoppable_nodes) < replicas:
            raise CliException(f"No more {node_size} nodes can be stopped")

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
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']
        resource_group = cluster_details['cloud']['resource_group']

        # Stop node
        AzureExecutor.stop_vm(
            resource_group=resource_group,
            vm_name=f"{cluster_id}-{node_name}-vm"
        )

        # Update node status
        self.grass_executor.remote_update_node_status(
            node_name=node_name,
            action='stop'
        )

        logger.info_green(f"Node {node_name} is stopped")

    def _get_node_size_to_spec(self) -> dict:
        # Load details
        cluster_details = self.cluster_details
        location = cluster_details['cloud']['location']

        # List available sizes for VMs
        specs = AzureExecutor.list_vm_sizes(location=location)

        # Get node_size_to_spec
        node_size_to_spec = {}
        for spec in specs:
            node_size_to_spec[spec['name']] = spec

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
        containers_details = node_details['containers']

        # Do counting
        count = 0
        for container_details in containers_details:
            if container_details['Status'] == 'running':
                count += 1

        return count

    # maro grass image

    def push_image(
        self, image_name: str, image_path: str, remote_context_path: str,
        remote_image_name: str
    ):
        # Load details
        cluster_details = self.cluster_details
        admin_username = cluster_details['user']['admin_username']
        master_public_ip_address = cluster_details['master']['public_ip_address']

        # Get images dir
        images_dir = f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/images"

        # Push image
        if image_name:
            new_file_name = get_valid_file_name(image_name)
            image_path = f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/images/{new_file_name}"
            self._save_image(
                image_name=image_name,
                export_path=os.path.expanduser(image_path)
            )
            if self._check_checksum_validity(
                local_file_path=os.path.expanduser(image_path),
                remote_file_path=os.path.join(images_dir, image_name)
            ):
                logger.info_green(f"The image file '{new_file_name}' already exists")
                return
            copy_files_to_node(
                local_path=image_path,
                remote_dir=images_dir,
                admin_username=admin_username, node_ip_address=master_public_ip_address
            )
            self.grass_executor.remote_update_image_files_details()
            self._batch_load_images()
            logger.info_green(f"Image {image_name} is loaded")
        elif image_path:
            file_name = os.path.basename(image_path)
            new_file_name = get_valid_file_name(file_name)
            image_path = f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/images/{new_file_name}"
            copy_and_rename(
                source_path=os.path.expanduser(image_path),
                target_dir=image_path
            )
            if self._check_checksum_validity(
                local_file_path=os.path.expanduser(image_path),
                remote_file_path=os.path.join(images_dir, new_file_name)
            ):
                logger.info_green(f"The image file '{new_file_name}' already exists")
                return
            copy_files_to_node(
                local_path=image_path,
                remote_dir=images_dir,
                admin_username=admin_username, node_ip_address=master_public_ip_address
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
            raise CliException("Invalid arguments")

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
            if node_details['state'] == 'Running':
                params.append([
                    node_name,
                    GlobalParams.PARALLELS,
                    node_details['public_ip_address']
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

    # maro grass job

    def start_job(self, deployment_path: str):
        # Load start_job_deployment
        with open(deployment_path, 'r') as fr:
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
        cluster_details = self.cluster_details
        admin_username = cluster_details['user']['admin_username']
        master_public_ip_address = cluster_details['master']['public_ip_address']
        job_name = job_details['name']

        # Sync mkdir
        sync_mkdir(
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs/{job_name}",
            admin_username=admin_username, node_ip_address=master_public_ip_address
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
            admin_username=admin_username, node_ip_address=master_public_ip_address
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

    def get_job_logs(self, job_name: str, export_dir: str = './'):
        # Load details
        cluster_details = self.cluster_details
        job_details = load_job_details(
            cluster_name=self.cluster_name,
            job_name=job_name
        )
        admin_username = cluster_details['user']['admin_username']
        master_public_ip_address = cluster_details['master']['public_ip_address']
        job_id = job_details['id']

        # Copy logs from master
        try:
            copy_files_from_node(
                local_dir=export_dir,
                remote_path=f"~/.maro/logs/{job_id}",
                admin_username=admin_username, node_ip_address=master_public_ip_address
            )
        except CommandError:
            logger.error_red("No logs have been created at this time")

    @staticmethod
    def _standardize_start_job_deployment(start_job_deployment: dict):
        # Validate grass-azure-start-job
        optional_key_to_value = {
            "root['tags']": {}
        }
        with open(os.path.expanduser(
                f'{GlobalPaths.MARO_GRASS_LIB}/deployments/internal/grass-azure-start-job.yml')) as fr:
            start_job_template = yaml.safe_load(fr)
        validate_and_fill_dict(
            template_dict=start_job_template,
            actual_dict=start_job_deployment,
            optional_key_to_value=optional_key_to_value
        )

        # Validate component
        with open(os.path.expanduser(f'{GlobalPaths.MARO_GRASS_LIB}/deployments/internal/component.yml'), 'r') as fr:
            start_job_component_template = yaml.safe_load(fr)
        components_details = start_job_deployment['components']
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
        job_details['id'] = generate_job_id()

        # Set component id
        for component, component_details in job_details['components'].items():
            component_details['id'] = generate_component_id()

        # Save details
        save_job_details(
            cluster_name=self.cluster_name,
            job_name=job_name,
            job_details=job_details
        )

    # maro grass schedule

    def start_schedule(self, deployment_path: str):
        # Load start_schedule_deployment
        with open(deployment_path, 'r') as fr:
            start_schedule_deployment = yaml.safe_load(fr)

        # Standardize start_schedule_deployment
        self._standardize_start_schedule_deployment(start_schedule_deployment=start_schedule_deployment)
        schedule_name = start_schedule_deployment['name']

        # Load details
        cluster_details = self.cluster_details
        admin_username = cluster_details['user']['admin_username']
        master_public_ip_address = cluster_details['master']['public_ip_address']

        # Sync mkdir
        sync_mkdir(
            remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/schedules/{schedule_name}",
            admin_username=admin_username, node_ip_address=master_public_ip_address
        )

        # Save schedule deployment
        save_schedule_details(
            cluster_name=self.cluster_name,
            schedule_name=schedule_name,
            schedule_details=start_schedule_deployment
        )

        # Start jobs
        for job_name in start_schedule_deployment['job_names']:
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
        job_names = schedule_details['job_names']

        for job_name in job_names:
            # Load job details
            job_details = load_job_details(cluster_name=self.cluster_name, job_name=job_name)
            job_schedule_tag = job_details['tags']['schedule']

            # Remote stop job
            if job_schedule_tag == schedule_name:
                self.grass_executor.remote_create_killed_job_ticket(job_name=job_name)
                self.grass_executor.remote_delete_pending_job_ticket(job_name=job_name)

    @staticmethod
    def _standardize_start_schedule_deployment(start_schedule_deployment: dict):
        # Validate grass-azure-start-job
        with open(os.path.expanduser(
                f'{GlobalPaths.MARO_GRASS_LIB}/deployments/internal/grass-azure-start-schedule.yml')) as fr:
            start_job_template = yaml.safe_load(fr)
        validate_and_fill_dict(
            template_dict=start_job_template,
            actual_dict=start_schedule_deployment,
            optional_key_to_value={}
        )

        # Validate component
        with open(os.path.expanduser(f'{GlobalPaths.MARO_GRASS_LIB}/deployments/internal/component.yml')) as fr:
            start_job_component_template = yaml.safe_load(fr)
        components_details = start_schedule_deployment['components']
        for _, component_details in components_details.items():
            validate_and_fill_dict(
                template_dict=start_job_component_template,
                actual_dict=component_details,
                optional_key_to_value={}
            )

    @staticmethod
    def _build_job_details(schedule_details: dict, job_name: str) -> dict:
        schedule_name = schedule_details['name']

        job_details = deepcopy(schedule_details)
        job_details['name'] = job_name
        job_details['tags'] = {
            'schedule': schedule_name
        }
        job_details.pop('job_names')

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
        else:
            raise CliException(f"Resource {resource_name} is unsupported")

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
        command = f'cp {GlobalPaths.MARO_GRASS_LIB}/deployments/external/* {export_path}'
        _ = SubProcess.run(command)
