# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import os
from copy import deepcopy
from shutil import rmtree

import yaml

from maro.cli.utils.copy import get_reformatted_source_path, get_reformatted_target_dir
from maro.cli.utils.details import (
    load_cluster_details, load_job_details, load_schedule_details, save_cluster_details, save_job_details,
    save_schedule_details
)
from maro.cli.utils.executors.azure_executor import AzureExecutor
from maro.cli.utils.naming import generate_cluster_id, generate_component_id, generate_job_id, generate_name_with_md5
from maro.cli.utils.params import GlobalPaths
from maro.cli.utils.subprocess import SubProcess
from maro.cli.utils.validation import validate_and_fill_dict
from maro.utils.exception.cli_exception import CliException
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class K8sAzureExecutor:

    def __init__(self, cluster_name: str):
        self.cluster_name = cluster_name
        self.cluster_details = load_cluster_details(cluster_name=cluster_name)

    # maro k8s create

    @staticmethod
    def build_cluster_details(create_deployment: dict):
        # Validate and fill optional value to deployment
        K8sAzureExecutor._validate_create_deployment(create_deployment=create_deployment)

        # Get cluster name and save details
        cluster_name = create_deployment['name']
        if os.path.isdir(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}")):
            raise CliException(f"cluster {cluster_name} is exist")
        os.makedirs(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}"))
        save_cluster_details(
            cluster_name=cluster_name,
            cluster_details=create_deployment
        )

    @staticmethod
    def _validate_create_deployment(create_deployment: dict):
        optional_key_to_value = {
            "root['master']['redis']": {'port': 6379},
            "root['master']['redis']['port']": 6379
        }
        with open(os.path.expanduser(
                f'{GlobalPaths.MARO_K8S_LIB}/deployments/internal/k8s-azure-create.yml')) as fr:
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
            self._create_k8s_cluster()
            self._init_redis()
            self._init_nvidia_plugin()
            self._create_k8s_secret()
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
        # Load details
        cluster_details = self.cluster_details
        subscription = cluster_details['cloud']['subscription']
        resource_group = cluster_details['cloud']['resource_group']
        location = cluster_details['cloud']['location']

        # Check if Azure CLI is installed, and print version
        azure_version = AzureExecutor.get_version()
        logger.info_green(f"Your Azure CLI version: {azure_version['azure-cli']}")

        # Set subscription id
        AzureExecutor.set_subscription(subscription=subscription)

        # Check and create resource group
        resource_group_info = AzureExecutor.get_resource_group(resource_group=resource_group)
        if resource_group_info is not None:
            logger.warning_yellow(f"Azure resource group {resource_group} is already existed")
        else:
            AzureExecutor.create_resource_group(
                resource_group=resource_group,
                location=location
            )
            logger.info_green(f"Resource group: {resource_group} is created")

    def _create_k8s_cluster(self):
        logger.info("Creating k8s cluster")

        # Load details
        cluster_details = self.cluster_details
        resource_group = cluster_details['cloud']['resource_group']

        # Create ARM parameters
        self._create_deployment_parameters(
            export_dir=os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/parameters")
        )

        # Start deployment
        template_file_location = f"{GlobalPaths.MARO_K8S_LIB}/azure/k8s-create-template.json"
        parameters_file_location = f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/parameters/aks_cluster.json"
        AzureExecutor.start_deployment(
            resource_group=resource_group,
            deployment_name='aks_cluster',
            template_file=template_file_location,
            parameters_file=parameters_file_location
        )

        # Attach ACR
        self._attach_acr()

    def _create_deployment_parameters(self, export_dir: str):
        # Extract variables
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']
        location = cluster_details['cloud']['location']
        admin_username = cluster_details['user']['admin_username']
        admin_public_key = cluster_details['user']['admin_public_key']
        node_size = cluster_details['master']['node_size']

        # Mkdir
        os.makedirs(export_dir, exist_ok=True)

        with open(os.path.expanduser(f"{GlobalPaths.MARO_K8S_LIB}/azure/k8s-create-parameters.json"), 'r') as f:
            base_parameters = json.load(f)
        with open(export_dir + "/aks_cluster.json", 'w') as fw:
            parameters = base_parameters['parameters']
            parameters['location']['value'] = location
            parameters['adminUsername']['value'] = admin_username
            parameters['adminPublicKey']['value'] = admin_public_key
            parameters['clusterName']['value'] = f"{cluster_id}-aks"
            parameters['agentCount']['value'] = 1
            parameters['agentVMSize']['value'] = node_size
            parameters['virtualNetworkName']['value'] = f"{cluster_id}-vnet"
            parameters['acrName']['value'] = f"{cluster_id}acr"
            parameters['acrSku']['value'] = "Basic"
            parameters['storageAccountName']['value'] = f"{cluster_id}st"
            parameters['fileShareName']['value'] = f"{cluster_id}-fs"
            json.dump(base_parameters, fw, indent=4)

    def _attach_acr(self):
        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']
        resource_group = cluster_details['cloud']['resource_group']

        # Attach ACR
        AzureExecutor.attach_acr(
            resource_group=resource_group,
            aks_name=f"{cluster_id}-aks",
            acr_name=f"{cluster_id}acr"
        )

    def _init_redis(self):
        # Create folder
        os.makedirs(os.path.expanduser(f'{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/k8s_configs'), exist_ok=True)

        # Create k8s config
        self._create_redis_k8s_config(
            export_dir=os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/k8s_configs")
        )

        # Apply k8s config
        command = f"kubectl apply -f {GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/k8s_configs/redis.yml"
        _ = SubProcess.run(command)

    @staticmethod
    def _init_nvidia_plugin():
        # Create plugin namespace
        command = "kubectl create namespace gpu-resources"
        _ = SubProcess.run(command)

        # Apply k8s config
        command = f"kubectl apply -f {GlobalPaths.MARO_K8S_LIB}/k8s_configs/nvidia-device-plugin.yml"
        _ = SubProcess.run(command)

    def _create_redis_k8s_config(self, export_dir: str):
        # Check and load k8s context
        self._check_and_load_k8s_context()

        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']

        # Fill redis k8s config and saves
        with open(os.path.expanduser(f"{GlobalPaths.MARO_K8S_LIB}/k8s_configs/redis.yml"), 'r') as fr:
            base_config = yaml.safe_load(fr)
        with open(export_dir + "/redis.yml", 'w') as fw:
            azure_file_config = base_config['spec']['template']['spec']['volumes'][0]['azureFile']
            azure_file_config['secretName'] = f"{cluster_id}-k8s-secret"
            azure_file_config['shareName'] = f"{cluster_id}-fs"
            yaml.safe_dump(base_config, fw)

    def _create_k8s_secret(self):
        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']
        resource_group = cluster_details['cloud']['resource_group']

        # Get storage account key
        storage_account_keys = AzureExecutor.get_storage_account_keys(
            resource_group=resource_group,
            storage_account_name=f"{cluster_id}st"
        )
        storage_key = storage_account_keys[0]['value']

        # Create k8s secret
        command = f'kubectl create secret generic {cluster_id}-k8s-secret ' \
            f'--from-literal=azurestorageaccountname={cluster_id}st ' \
            f'--from-literal=azurestorageaccountkey={storage_key}'
        _ = SubProcess.run(command)
        logger.debug(command)

    # maro k8s delete

    def delete(self):
        logger.info(f"Deleting cluster {self.cluster_name}")

        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']
        resource_group = cluster_details['cloud']['resource_group']

        # Get resource list
        resource_list = AzureExecutor.list_resources(resource_group=resource_group)

        # Filter resources
        deletable_ids = []
        for resource in resource_list:
            if resource['name'].startswith(cluster_id):
                deletable_ids.append(resource['id'])

        # Delete resources
        if deletable_ids:
            AzureExecutor.delete_resources(resources=deletable_ids)

        # Delete cluster folder
        rmtree(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}"))

        logger.info_green(f"Cluster {self.cluster_name} is deleted")

    # maro k8s node

    def scale_node(self, replicas: int, node_size: str):
        # Get node_size_to_info
        node_size_to_info = self._get_node_size_to_info()

        # Get node_size_to_spec, and check if node_size is valid
        node_size_to_spec = self._get_node_size_to_spec()
        if node_size not in node_size_to_spec:
            raise CliException(f"Invalid node_size: {node_size}")

        # Scale node
        if node_size not in node_size_to_info:
            self._build_node_pool(
                replicas=replicas,
                node_size=node_size
            )
        elif node_size_to_info[node_size]['count'] != replicas:
            self._scale_node_pool(
                replicas=replicas,
                node_size=node_size,
                node_size_to_info=node_size_to_info
            )
        else:
            logger.warning_yellow("Replica is match, no create or delete")

    def _get_node_size_to_info(self):
        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']
        resource_group = cluster_details['cloud']['resource_group']

        # List nodepool
        nodepools = AzureExecutor.list_nodepool(
            resource_group=resource_group,
            aks_name=f"{cluster_id}-aks"
        )

        # Build node_size_to_count
        node_size_to_count = {}
        for nodepool in nodepools:
            node_size_to_count[nodepool['vmSize']] = nodepool

        return node_size_to_count

    def _get_node_size_to_spec(self) -> dict:
        # Load details
        cluster_details = self.cluster_details
        location = cluster_details['cloud']['location']

        # List available sizes for VM
        specs = AzureExecutor.list_vm_sizes(location=location)

        # Build node_size_to_spec
        node_size_to_spec = {}
        for spec in specs:
            node_size_to_spec[spec['name']] = spec

        return node_size_to_spec

    def _build_node_pool(self, replicas: int, node_size: str):
        logger.info(f"Building {node_size} NodePool")

        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']
        resource_group = cluster_details['cloud']['resource_group']

        # Build nodepool
        AzureExecutor.add_nodepool(
            resource_group=resource_group,
            aks_name=f"{cluster_id}-aks",
            nodepool_name=K8sAzureExecutor._generate_nodepool_name(key=node_size),
            node_count=replicas,
            node_size=node_size
        )

        logger.info_green(f"{node_size} NodePool is built")

    def _scale_node_pool(self, replicas: int, node_size: str, node_size_to_info: dict):
        logger.info(f"Scaling {node_size} NodePool")

        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']
        resource_group = cluster_details['cloud']['resource_group']

        # Scale node pool
        AzureExecutor.scale_nodepool(
            resource_group=resource_group,
            aks_name=f"{cluster_id}-aks",
            nodepool_name=node_size_to_info[node_size]['name'],
            node_count=replicas
        )

        logger.info_green(f"{node_size} NodePool is scaled")

    @staticmethod
    def _generate_nodepool_name(key: str) -> str:
        return generate_name_with_md5(prefix='pool', key=key, md5_len=8)

    def list_node(self):
        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']
        resource_group = cluster_details['cloud']['resource_group']

        # Get aks details
        aks_details = AzureExecutor.get_aks(
            resource_group=resource_group,
            aks_name=f"{cluster_id}-aks"
        )
        agent_pools_details = aks_details['agentPoolProfiles']

        # Filter and print
        node_details = {}
        for agent_pool_details in agent_pools_details:
            node_details[agent_pool_details['vmSize']] = agent_pool_details['count']
        logger.info(
            json.dumps(
                node_details,
                indent=4, sort_keys=True
            )
        )

    # maro k8s image

    def push_image(self, image_name: str):
        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']
        remote_image_name = f"{cluster_id}acr.azurecr.io/{image_name}"

        # ACR login
        AzureExecutor.login_acr(acr_name=f"{cluster_id}acr")

        # Tag image
        command = f"docker tag {image_name} {remote_image_name}"
        _ = SubProcess.run(command)

        # Push image to ACR
        command = f"docker push {remote_image_name}"
        _ = SubProcess.run(command)

    def list_image(self):
        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']

        # List acr repository
        acr_repositories = AzureExecutor.list_acr_repositories(acr_name=f"{cluster_id}acr")
        logger.info(acr_repositories)

    # maro k8s data

    def push_data(self, local_path: str, remote_dir: str):
        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']

        # Get sas
        sas = self._check_and_get_account_sas()

        # Push data
        local_path = os.path.expanduser(local_path)
        source_path = get_reformatted_source_path(local_path)
        target_dir = get_reformatted_target_dir(remote_dir)
        if not target_dir.startswith("/"):
            raise CliException("Invalid remote path")
        copy_command = f'azcopy copy ' \
            f'"{source_path}" ' \
            f'"https://{cluster_id}st.file.core.windows.net/{cluster_id}-fs{target_dir}?{sas}" ' \
            f'--recursive=True'
        _ = SubProcess.run(copy_command)

    def pull_data(self, local_dir: str, remote_path: str):
        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']

        # Get sas
        sas = self._check_and_get_account_sas()

        # Push data
        local_dir = os.path.expanduser(local_dir)
        source_path = get_reformatted_source_path(remote_path)
        target_dir = get_reformatted_target_dir(local_dir)
        mkdir_script = f"mkdir -p {target_dir}"
        _ = SubProcess.run(mkdir_script)
        if not source_path.startswith("/"):
            raise CliException("Invalid remote path")
        copy_command = f'azcopy copy ' \
            f'"https://{cluster_id}st.file.core.windows.net/{cluster_id}-fs{source_path}?{sas}" ' \
            f'"{os.path.expanduser(target_dir)}" ' \
            f'--recursive=True'
        _ = SubProcess.run(copy_command)

    def remove_data(self, remote_path: str):
        # FIXME: Remove failed, The specified resource may be in use by an SMB client

        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']

        # Get sas
        sas = self._check_and_get_account_sas()

        # Remove data
        copy_command = f'azcopy remove ' \
            f'"https://{cluster_id}st.file.core.windows.net/{cluster_id}-fs{remote_path}?{sas}" ' \
            f'--recursive=True'
        _ = SubProcess.run(copy_command)

    def _check_and_get_account_sas(self):
        """
        Ref: https://msdn.microsoft.com/library/azure/mt584140.aspx
        """

        # Load details
        cluster_details = self.cluster_details
        cloud_details = cluster_details['cloud']
        cluster_id = cluster_details['id']

        # Regenerate sas if the key is None or expired TODO:
        if 'account_sas' not in cloud_details:
            account_sas = AzureExecutor.get_storage_account_sas(account_name=f'{cluster_id}st')
            cloud_details['account_sas'] = account_sas
            save_cluster_details(
                cluster_name=self.cluster_name,
                cluster_details=cluster_details
            )

        return cloud_details['account_sas']

    # maro k8s job

    def start_job(self, deployment_path: str):
        # Load start_job_deployment
        with open(deployment_path, 'r') as fr:
            start_job_deployment = yaml.safe_load(fr)

        # Start job
        self._start_job(job_details=start_job_deployment)

    def _start_job(self, job_details: dict):
        # Validate and fill optional value to deployment
        K8sAzureExecutor._standardize_start_job_deployment(start_job_deployment=job_details)
        job_name = job_details['name']

        # Mkdir and save job details
        os.makedirs(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs/{job_name}"),
                    exist_ok=True)
        save_job_details(
            cluster_name=self.cluster_name,
            job_name=job_name,
            job_details=job_details
        )

        # Set job id
        self._set_job_id(job_name=job_name)

        # Create folder
        os.makedirs(os.path.expanduser(
            f'{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs/{job_name}/k8s_configs'), exist_ok=True)

        # Create and save k8s config
        k8s_job_config = self._create_k8s_job_config(job_name=job_name)
        with open(os.path.expanduser(
                f'{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs/{job_name}/k8s_configs/jobs.yml'), 'w') as fw:
            yaml.safe_dump(k8s_job_config, fw)

        # Apply k8s config
        command = f"kubectl apply -f " \
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs/{job_name}/k8s_configs/jobs.yml"
        _ = SubProcess.run(command)

    def stop_job(self, job_name: str):
        # Stop job
        command = f"kubectl delete -f " \
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/jobs/{job_name}/k8s_configs/jobs.yml"
        _ = SubProcess.run(command)

    @staticmethod
    def _standardize_start_job_deployment(start_job_deployment: dict):
        # Validate k8s-azure-start-job
        with open(os.path.expanduser(
                f'{GlobalPaths.MARO_K8S_LIB}/deployments/internal/k8s-azure-start-job.yml')) as fr:
            start_job_template = yaml.safe_load(fr)
        validate_and_fill_dict(
            template_dict=start_job_template,
            actual_dict=start_job_deployment,
            optional_key_to_value={}
        )

        # Validate component
        with open(os.path.expanduser(f'{GlobalPaths.MARO_K8S_LIB}/deployments/internal/component.yml'), 'r') as fr:
            component_template = yaml.safe_load(fr)
        components_details = start_job_deployment['components']
        for _, component_details in components_details.items():
            validate_and_fill_dict(
                template_dict=component_template,
                actual_dict=component_details,
                optional_key_to_value={}
            )

    def _create_k8s_job_config(self, job_name: str) -> dict:
        # Load details
        cluster_details = self.cluster_details
        job_details = load_job_details(cluster_name=self.cluster_name, job_name=job_name)
        cluster_id = cluster_details['id']
        job_id = job_details["id"]

        # Check and load k8s context
        self._check_and_load_k8s_context()

        # Get config template
        with open(os.path.expanduser(f"{GlobalPaths.MARO_K8S_LIB}/k8s_configs/job.yml")) as fr:
            k8s_job_config = yaml.safe_load(fr)
        with open(os.path.expanduser(f"{GlobalPaths.MARO_K8S_LIB}/k8s_configs/container.yml")) as fr:
            k8s_container_config = yaml.safe_load(fr)

        # Fill configs
        k8s_job_config['metadata']['name'] = job_id
        k8s_job_config['metadata']['labels']['jobName'] = job_name
        azure_file_config = k8s_job_config['spec']['template']['spec']['volumes'][0]['azureFile']
        azure_file_config['secretName'] = f"{cluster_id}-k8s-secret"
        azure_file_config['shareName'] = f"{cluster_id}-fs"

        # Create and fill container config
        for component_type, component_details in job_details['components'].items():
            for component_index in range(component_details['num']):
                container_config = self._create_k8s_container_config(
                    job_details=job_details,
                    k8s_container_config_template=k8s_container_config,
                    component_type=component_type,
                    component_index=component_index
                )
                k8s_job_config['spec']['template']['spec']['containers'].append(container_config)

        return k8s_job_config

    def _create_k8s_container_config(
        self, job_details: dict, k8s_container_config_template: dict,
        component_type: str, component_index: int
    ):
        # Copy config
        k8s_container_config = deepcopy(k8s_container_config_template)

        # Get container config
        component_details = job_details['components'][component_type]

        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']
        job_name = job_details['name']
        job_id = job_details['id']
        component_id = job_details["components"][component_type]['id']
        container_name = f"{job_id}-{component_id}-{component_index}"

        # Fill config
        k8s_container_config['name'] = container_name
        k8s_container_config['image'] = self._build_image_address(image_name=component_details['image'])
        k8s_container_config['resources']['requests'] = {
            'cpu': component_details['resources']['cpu'],
            'memory': component_details['resources']['memory'],
            'nvidia.com/gpu': component_details['resources']['gpu']
        }
        k8s_container_config['resources']['limits'] = {
            'cpu': component_details['resources']['cpu'],
            'memory': component_details['resources']['memory'],
            'nvidia.com/gpu': component_details['resources']['gpu']
        }
        k8s_container_config['env'] = [
            {
                'name': 'COMPONENT_TYPE',
                'value': f"{component_type}"
            },
            {
                'name': 'COMPONENT_ID',
                'value': f"{component_id}"
            },
            {
                'name': 'COMPONENT_INDEX',
                'value': f"{component_index}"
            },
            {
                'name': 'JOB_NAME',
                'value': f"{job_name}"
            },
            {
                'name': 'JOB_ID',
                'value': f"{job_id}"
            },
            {
                'name': 'CLUSTER_NAME',
                'value': f"{self.cluster_name}"
            },
            {
                'name': 'CLUSTER_ID',
                'value': f"{cluster_id}"
            },
            {
                'name': 'PYTHONUNBUFFERED',
                'value': '0'
            }
        ]
        k8s_container_config['command'] = component_details['command']
        k8s_container_config['volumeMounts'][0]['mountPath'] = component_details['mount']['target']

        return k8s_container_config

    def _set_job_id(self, job_name: str):
        # Load job details
        job_details = load_job_details(
            cluster_name=self.cluster_name,
            job_name=job_name
        )

        # Set job id
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

    def _build_image_address(self, image_name: str) -> str:
        # Load details
        cluster_id = self.cluster_details['id']

        # Get repositories
        acr_repositories = AzureExecutor.list_acr_repositories(acr_name=f"{cluster_id}acr")

        # Build address
        if image_name in acr_repositories:
            return f"{cluster_id}acr.azurecr.io/{image_name}"
        else:
            return image_name

    def list_job(self):
        # Get jobs details
        command = "kubectl get jobs -o=json"
        return_str = SubProcess.run(command)
        job_details_list = json.loads(return_str)["items"]
        jobs_details = {}
        for job_details in job_details_list:
            jobs_details[job_details["metadata"]["labels"]["jobName"]] = job_details

        # Print details
        logger.info(
            json.dumps(
                jobs_details,
                indent=4, sort_keys=True
            )
        )

    def get_job_logs(self, job_name: str, export_dir: str = './'):
        # Load details
        job_details = load_job_details(cluster_name=self.cluster_name, job_name=job_name)
        job_id = job_details['id']

        # Get pods details
        pods_details = self.get_pods_details()

        # Export logs
        for pod_details in pods_details:
            if pod_details["metadata"]["name"].startswith(job_id):
                for container_details in pod_details["spec"]["containers"]:
                    self._export_log(
                        pod_id=pod_details["metadata"]["name"],
                        container_name=container_details["name"],
                        export_dir=export_dir
                    )

    @staticmethod
    def _export_log(pod_id: str, container_name: str, export_dir: str):
        os.makedirs(os.path.expanduser(export_dir + f"/{pod_id}"), exist_ok=True)
        with open(os.path.expanduser(export_dir + f"/{pod_id}/{container_name}.log"), "w") as fw:
            command = f"kubectl logs {pod_id} {container_name}"
            return_str = SubProcess.run(command)
            fw.write(return_str)

    # maro k8s schedule

    def start_schedule(self, deployment_path: str):
        # Load start_schedule_deployment
        with open(deployment_path, 'r') as fr:
            start_schedule_deployment = yaml.safe_load(fr)

        # Standardize start_schedule_deployment
        K8sAzureExecutor._standardize_start_schedule_deployment(start_schedule_deployment=start_schedule_deployment)
        schedule_name = start_schedule_deployment['name']

        # Save schedule deployment
        os.makedirs(os.path.expanduser(
            f"{GlobalPaths.MARO_CLUSTERS}/{self.cluster_name}/schedules/{schedule_name}"), exist_ok=True)
        save_schedule_details(
            cluster_name=self.cluster_name,
            schedule_name=schedule_name,
            schedule_details=start_schedule_deployment
        )

        # Start jobs
        for job_name in start_schedule_deployment['job_names']:
            job_details = K8sAzureExecutor._build_job_details(
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

            # Stop job
            if job_schedule_tag == schedule_name:
                self.stop_job(
                    job_name=job_name
                )

    @staticmethod
    def _standardize_start_schedule_deployment(start_schedule_deployment: dict):
        # Validate k8s-azure-start-job
        with open(os.path.expanduser(
                f'{GlobalPaths.MARO_K8S_LIB}/deployments/internal/k8s-azure-start-schedule.yml')) as fr:
            start_job_template = yaml.safe_load(fr)
        validate_and_fill_dict(
            template_dict=start_job_template,
            actual_dict=start_schedule_deployment,
            optional_key_to_value={}
        )

        # Validate component
        with open(os.path.expanduser(f'{GlobalPaths.MARO_K8S_LIB}/deployments/internal/component.yml')) as fr:
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
        job_details['tags'] = {'schedule': schedule_name}
        job_details.pop('job_names')

        return job_details

    # maro k8s status

    def status(self):
        return_status = {}

        # Get pods details
        pods_details = self.get_pods_details()

        for pod_details in pods_details:
            if "app" in pod_details['metadata']['labels'] and pod_details['metadata']['labels']['app'] == 'maro-redis':
                return_status['redis'] = {
                    'private_ip_address': pod_details['status']['podIP']
                }
                break

        # Print status
        logger.info(
            json.dumps(
                return_status,
                indent=4, sort_keys=True
            )
        )

    # maro k8s template

    @staticmethod
    def template(export_path: str):
        command = f'cp {GlobalPaths.MARO_K8S_LIB}/deployments/external/* {export_path}'
        _ = SubProcess.run(command)

    # utils

    def _load_k8s_context(self):
        # Load details
        cluster_details = self.cluster_details
        resource_group = cluster_details['cloud']['resource_group']
        cluster_id = cluster_details['id']

        # Load context
        AzureExecutor.load_aks_context(
            resource_group=resource_group,
            aks_name=f"{cluster_id}-aks"
        )

    def _check_and_load_k8s_context(self):
        # Load details
        cluster_details = self.cluster_details
        cluster_id = cluster_details['id']

        # Check and load k8s context
        check_command = 'kubectl config view'
        config_str = SubProcess.run(check_command)
        config_dict = yaml.safe_load(config_str)
        if config_dict['current-context'] != f"{cluster_id}-aks":
            self._load_k8s_context()

    @staticmethod
    def get_pods_details():
        # Get pods details
        command = "kubectl get pods -o json"
        return_str = SubProcess.run(command)
        return json.loads(return_str)['items']
