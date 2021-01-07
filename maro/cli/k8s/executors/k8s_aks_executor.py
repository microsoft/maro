# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import base64
import copy
import json
import os
import shutil

import yaml
from kubernetes import client, config

from maro.cli.k8s.executors.k8s_executor import K8sExecutor
from maro.cli.utils.azure_controller import AzureController
from maro.cli.utils.deployment_validator import DeploymentValidator
from maro.cli.utils.details_reader import DetailsReader
from maro.cli.utils.details_writer import DetailsWriter
from maro.cli.utils.name_creator import NameCreator
from maro.cli.utils.params import GlobalPaths, GlobalParams
from maro.cli.utils.path_convertor import PathConvertor
from maro.cli.utils.subprocess import SubProcess
from maro.utils.exception.cli_exception import BadRequestError, FileOperationError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class K8sAksExecutor(K8sExecutor):

    def __init__(self, cluster_name: str):
        self.cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

        # Cloud configs
        self.subscription = self.cluster_details["cloud"]["subscription"]
        self.resource_group = self.cluster_details["cloud"]["resource_group"]
        self.location = self.cluster_details["cloud"]["location"]

        super().__init__(cluster_details=self.cluster_details)

    # maro k8s create

    @staticmethod
    def create(create_deployment: dict):
        logger.info("Creating cluster")

        # Get standardized cluster_details
        cluster_details = K8sAksExecutor._standardize_cluster_details(create_deployment=create_deployment)
        cluster_name = cluster_details["name"]
        cluster_id = cluster_details["id"]
        resource_group = cluster_details["cloud"]["resource_group"]
        if os.path.isdir(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}"):
            raise BadRequestError(f"Cluster '{cluster_name}' is exist")

        # Start creating
        try:
            K8sAksExecutor._create_resource_group(cluster_details=cluster_details)
            K8sAksExecutor._create_k8s_cluster(cluster_details=cluster_details)
            K8sAksExecutor._load_k8s_context(cluster_id=cluster_id, resource_group=resource_group)
            K8sAksExecutor._init_redis()
            K8sAksExecutor._init_nvidia_plugin()
            K8sAksExecutor._create_storage_account_secret(cluster_details=cluster_details)
            DetailsWriter.save_cluster_details(cluster_name=cluster_name, cluster_details=cluster_details)
        except Exception as e:
            # If failed, remove details folder, then raise
            shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}")
            logger.error_red(f"Failed to create cluster '{cluster_name}'")
            raise e

        logger.info_green(f"Cluster '{cluster_name}' is created")

    @staticmethod
    def _standardize_cluster_details(create_deployment: dict):
        optional_key_to_value = {
            "root['master']['redis']": {
                "port": GlobalParams.DEFAULT_REDIS_PORT
            },
            "root['master']['redis']['port']": GlobalParams.DEFAULT_REDIS_PORT
        }
        with open(f"{GlobalPaths.ABS_MARO_K8S_LIB}/deployments/internal/k8s_aks_create.yml") as fr:
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
    def _create_resource_group(cluster_details: dict):
        # Load details
        subscription = cluster_details["cloud"]["subscription"]
        resource_group = cluster_details["cloud"]["resource_group"]
        location = cluster_details["cloud"]["location"]

        # Check if Azure CLI is installed, and print version
        azure_version = AzureController.get_version()
        logger.info_green(f"Your Azure CLI version: {azure_version['azure-cli']}")

        # Set subscription id
        AzureController.set_subscription(subscription=subscription)
        logger.info_green(f"Set subscription to '{subscription}'")

        # Check and create resource group
        resource_group_info = AzureController.get_resource_group(resource_group=resource_group)
        if resource_group_info is not None:
            logger.warning_yellow(f"Azure resource group '{resource_group}' already exists")
        else:
            AzureController.create_resource_group(
                resource_group=resource_group,
                location=location
            )
            logger.info_green(f"Resource group '{resource_group}' is created")

    @staticmethod
    def _create_k8s_cluster(cluster_details: dict):
        logger.info("Creating k8s cluster")

        # Create ARM parameters and start deployment
        template_file_path = f"{GlobalPaths.ABS_MARO_K8S_LIB}/clouds/aks/create_aks_cluster/template.json"
        parameters_file_path = (
            f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_details['name']}/parameters/create_aks_cluster.json"
        )
        ArmTemplateParameterBuilder.create_aks_cluster(
            cluster_details=cluster_details,
            export_path=parameters_file_path
        )
        AzureController.start_deployment(
            resource_group=cluster_details["cloud"]["resource_group"],
            deployment_name="aks_cluster",
            template_file_path=template_file_path,
            parameters_file_path=parameters_file_path
        )

        # Attach ACR
        K8sAksExecutor._attach_acr(cluster_details=cluster_details)

        logger.info_green("K8s cluster is created")

    @staticmethod
    def _attach_acr(cluster_details: dict):
        # Load details
        cluster_id = cluster_details["id"]
        resource_group = cluster_details["cloud"]["resource_group"]

        # Attach ACR
        AzureController.attach_acr(
            resource_group=resource_group,
            aks_name=f"{cluster_id}-aks",
            acr_name=f"{cluster_id}acr"
        )

    @staticmethod
    def _init_nvidia_plugin():
        client.CoreV1Api().create_namespace(body=client.V1Namespace(metadata=client.V1ObjectMeta(name="gpu-resources")))

        with open(
            f"{GlobalPaths.ABS_MARO_K8S_LIB}/clouds/aks/create_nvidia_plugin/nvidia-device-plugin.yml", "r"
        ) as fr:
            redis_deployment = yaml.safe_load(fr)
        client.AppsV1Api().create_namespaced_daemon_set(body=redis_deployment, namespace="gpu-resources")

    @staticmethod
    def _create_storage_account_secret(cluster_details: dict):
        # Load details
        cluster_id = cluster_details["id"]
        resource_group = cluster_details["cloud"]["resource_group"]

        # Get storage account key
        storage_account_keys = AzureController.get_storage_account_keys(
            resource_group=resource_group,
            storage_account_name=f"{cluster_id}st"
        )
        storage_key = storage_account_keys[0]["value"]

        # Create k8s secret
        client.CoreV1Api().create_namespaced_secret(
            body=client.V1Secret(
                metadata=client.V1ObjectMeta(name="azure-storage-account-secret"),
                data={
                    "azurestorageaccountname": base64.b64encode(f"{cluster_id}st".encode()).decode(),
                    "azurestorageaccountkey": base64.b64encode(bytes(storage_key.encode())).decode()
                }
            ),
            namespace="default"
        )

    # maro k8s delete

    def delete(self):
        logger.info(f"Deleting cluster '{self.cluster_name}'")

        # Get resource list
        resource_list = AzureController.list_resources(resource_group=self.resource_group)

        # Filter resources
        deletable_ids = []
        for resource in resource_list:
            if resource["name"].startswith(self.cluster_id):
                deletable_ids.append(resource["id"])

        # Delete resources
        if deletable_ids:
            AzureController.delete_resources(resources=deletable_ids)

        # Delete cluster folder
        shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}")

        logger.info_green(f"Cluster '{self.cluster_name}' is deleted")

    # maro k8s node

    def scale_node(self, replicas: int, node_size: str):
        # Get node_size_to_info
        node_size_to_info = self._get_node_size_to_info()

        # Get node_size_to_spec, and check if node_size is valid
        node_size_to_spec = self._get_node_size_to_spec()
        if node_size not in node_size_to_spec:
            raise BadRequestError(f"Invalid node_size '{node_size}'")

        # Scale node
        if node_size not in node_size_to_info:
            self._build_node_pool(
                replicas=replicas,
                node_size=node_size
            )
        elif node_size_to_info[node_size]["count"] != replicas:
            self._scale_node_pool(
                replicas=replicas,
                node_size=node_size,
                node_size_to_info=node_size_to_info
            )
        else:
            logger.warning_yellow("Replica is match, no create or delete")

    def _get_node_size_to_info(self):
        # List nodepool
        nodepools = AzureController.list_nodepool(
            resource_group=self.resource_group,
            aks_name=f"{self.cluster_id}-aks"
        )

        # Build node_size_to_count
        node_size_to_count = {}
        for nodepool in nodepools:
            node_size_to_count[nodepool["vmSize"]] = nodepool

        return node_size_to_count

    def _get_node_size_to_spec(self) -> dict:
        # List available sizes for VM
        specs = AzureController.list_vm_sizes(location=self.location)

        # Build node_size_to_spec
        node_size_to_spec = {}
        for spec in specs:
            node_size_to_spec[spec["name"]] = spec

        return node_size_to_spec

    def _build_node_pool(self, replicas: int, node_size: str):
        logger.info(f"Building '{node_size}' nodepool")

        # Build nodepool
        AzureController.add_nodepool(
            resource_group=self.resource_group,
            aks_name=f"{self.cluster_id}-aks",
            nodepool_name=K8sAksExecutor._generate_nodepool_name(key=node_size),
            node_count=replicas,
            node_size=node_size
        )

        logger.info_green(f"'{node_size}' nodepool is built")

    def _scale_node_pool(self, replicas: int, node_size: str, node_size_to_info: dict):
        logger.info(f"Scaling '{node_size}' nodepool")

        # Scale node pool
        AzureController.scale_nodepool(
            resource_group=self.resource_group,
            aks_name=f"{self.cluster_id}-aks",
            nodepool_name=node_size_to_info[node_size]["name"],
            node_count=replicas
        )

        logger.info_green(f"'{node_size}' nodepool is scaled")

    @staticmethod
    def _generate_nodepool_name(key: str) -> str:
        return NameCreator.create_name_with_md5(prefix="pool", key=key, md5_len=8)

    def list_node(self):
        # Get aks details
        aks_details = AzureController.get_aks(
            resource_group=self.resource_group,
            aks_name=f"{self.cluster_id}-aks"
        )
        agent_pools_details = aks_details["agentPoolProfiles"]

        # Filter and print
        node_details = {}
        for agent_pool_details in agent_pools_details:
            node_details[agent_pool_details["vmSize"]] = agent_pool_details["count"]
        logger.info(
            json.dumps(
                node_details,
                indent=4, sort_keys=True
            )
        )

    # maro k8s image

    def push_image(self, image_name: str):
        remote_image_name = f"{self.cluster_id}acr.azurecr.io/{image_name}"

        # ACR login
        AzureController.login_acr(acr_name=f"{self.cluster_id}acr")

        # Tag image
        command = f"docker tag {image_name} {remote_image_name}"
        _ = SubProcess.run(command)

        # Push image to ACR
        command = f"docker push {remote_image_name}"
        _ = SubProcess.run(command)

    def list_image(self):
        # List acr repository
        acr_repositories = AzureController.list_acr_repositories(acr_name=f"{self.cluster_id}acr")
        logger.info(acr_repositories)

    # maro k8s data

    def push_data(self, local_path: str, remote_dir: str):
        # Get sas
        sas = self._check_and_get_account_sas()

        # Push data
        abs_local_path = os.path.expanduser(local_path)
        abs_source_path = PathConvertor.build_path_without_trailing_slash(abs_local_path)
        target_dir = PathConvertor.build_path_with_trailing_slash(remote_dir)
        if not target_dir.startswith("/"):
            raise FileOperationError(f"Invalid remote path: {target_dir}\nShould be started with '/'")
        copy_command = (
            "azcopy copy "
            f"'{abs_source_path}' "
            f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs{target_dir}?{sas}' "
            "--recursive=True"
        )
        _ = SubProcess.run(copy_command)

    def pull_data(self, local_dir: str, remote_path: str):
        # Get sas
        sas = self._check_and_get_account_sas()

        # Push data
        abs_local_dir = os.path.expanduser(local_dir)
        source_path = PathConvertor.build_path_without_trailing_slash(remote_path)
        abs_target_dir = PathConvertor.build_path_with_trailing_slash(abs_local_dir)
        os.makedirs(abs_target_dir, exist_ok=True)
        if not source_path.startswith("/"):
            raise FileOperationError(f"Invalid remote path: {source_path}\nShould be started with '/'")
        copy_command = (
            "azcopy copy "
            f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs{source_path}?{sas}' "
            f"'{abs_target_dir}' "
            "--recursive=True"
        )
        _ = SubProcess.run(copy_command)

    def remove_data(self, remote_path: str):
        # FIXME: Remove failed, The specified resource may be in use by an SMB client

        # Get sas
        sas = self._check_and_get_account_sas()

        # Remove data
        copy_command = (
            "azcopy remove "
            f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs{remote_path}?{sas}' "
            "--recursive=True"
        )
        _ = SubProcess.run(copy_command)

    def _check_and_get_account_sas(self):
        """
        Ref: https://msdn.microsoft.com/library/azure/mt584140.aspx
        """

        # Load details
        cloud_details = self.cluster_details["cloud"]

        # Regenerate sas if the key is None or expired TODO:
        if "account_sas" not in cloud_details:
            account_sas = AzureController.get_storage_account_sas(account_name=f"{self.cluster_id}st")
            cloud_details["account_sas"] = account_sas
            DetailsWriter.save_cluster_details(
                cluster_name=self.cluster_name,
                cluster_details=self.cluster_details
            )

        return cloud_details["account_sas"]

    # maro k8s job

    def _create_k8s_job(self, job_details: dict) -> dict:
        # Load details
        job_name = job_details["name"]
        job_id = job_details["id"]

        # Get config template
        with open(f"{GlobalPaths.ABS_MARO_K8S_LIB}/clouds/aks/create_job/job.yml") as fr:
            k8s_job_config = yaml.safe_load(fr)
        with open(f"{GlobalPaths.ABS_MARO_K8S_LIB}/clouds/aks/create_job/container.yml") as fr:
            k8s_container_config = yaml.safe_load(fr)

        # Fill configs
        k8s_job_config["metadata"]["name"] = job_id
        k8s_job_config["metadata"]["labels"]["jobName"] = job_name
        azure_file_config = k8s_job_config["spec"]["template"]["spec"]["volumes"][0]["azureFile"]
        azure_file_config["secretName"] = f"azure-storage-account-secret"
        azure_file_config["shareName"] = f"{self.cluster_id}-fs"

        # Create and fill container config
        for component_type, component_details in job_details["components"].items():
            for component_index in range(component_details["num"]):
                container_config = self._create_k8s_container_config(
                    job_details=job_details,
                    k8s_container_config_template=k8s_container_config,
                    component_type=component_type,
                    component_index=component_index
                )
                k8s_job_config["spec"]["template"]["spec"]["containers"].append(container_config)

        return k8s_job_config

    def _create_k8s_container_config(
        self, job_details: dict, k8s_container_config_template: dict,
        component_type: str, component_index: int
    ):
        # Copy config.
        k8s_container_config = copy.deepcopy(k8s_container_config_template)

        # Load details
        component_details = job_details["components"][component_type]
        job_id = job_details["id"]
        component_id = job_details["components"][component_type]["id"]
        container_name = f"{job_id}-{component_id}-{component_index}"

        # Fill configs.
        k8s_container_config["name"] = container_name
        k8s_container_config["image"] = self._build_image_address(image_name=component_details["image"])
        k8s_container_config["resources"]["requests"] = {
            "cpu": component_details["resources"]["cpu"],
            "memory": component_details["resources"]["memory"],
            "nvidia.com/gpu": component_details["resources"]["gpu"]
        }
        k8s_container_config["resources"]["limits"] = {
            "cpu": component_details["resources"]["cpu"],
            "memory": component_details["resources"]["memory"],
            "nvidia.com/gpu": component_details["resources"]["gpu"]
        }
        k8s_container_config["env"] = [
            {
                "name": "CLUSTER_ID",
                "value": f"{self.cluster_id}"
            },
            {
                "name": "CLUSTER_NAME",
                "value": f"{self.cluster_name}"
            },
            {
                "name": "JOB_ID",
                "value": f"{job_id}"
            },
            {
                "name": "JOB_NAME",
                "value": job_details["name"]
            },
            {
                "name": "COMPONENT_ID",
                "value": f"{component_id}"
            },
            {
                "name": "COMPONENT_TYPE",
                "value": f"{component_type}"
            },
            {
                "name": "COMPONENT_INDEX",
                "value": f"{component_index}"
            },
            {
                "name": "PYTHONUNBUFFERED",
                "value": "0"
            }
        ]
        k8s_container_config["command"] = component_details["command"]
        k8s_container_config["volumeMounts"][0]["mountPath"] = component_details["mount"]["target"]

        return k8s_container_config

    def _build_image_address(self, image_name: str) -> str:
        # Get repositories
        acr_repositories = AzureController.list_acr_repositories(acr_name=f"{self.cluster_id}acr")

        # Build address
        if image_name in acr_repositories:
            return f"{self.cluster_id}acr.azurecr.io/{image_name}"
        else:
            return image_name

    @staticmethod
    def _export_log(pod_id: str, container_name: str, export_dir: str):
        os.makedirs(os.path.expanduser(export_dir + f"/{pod_id}"), exist_ok=True)
        with open(os.path.expanduser(export_dir + f"/{pod_id}/{container_name}.log"), "w") as fw:
            return_str = client.CoreV1Api().read_namespaced_pod_log(name=pod_id, namespace="default")
            fw.write(return_str)

    # maro k8s status

    def status(self):
        return_status = {}

        # Get pods details
        pod_list = client.CoreV1Api().list_pod_for_all_namespaces(watch=False).to_dict()["items"]

        for pod in pod_list:
            if "app" in pod["metadata"]["labels"] and pod["metadata"]["labels"]["app"] == "maro-redis":
                return_status["redis"] = {
                    "private_ip_address": pod["status"]["pod_ip"]
                }
                break

        # Print status
        logger.info(
            json.dumps(
                return_status,
                indent=4, sort_keys=True
            )
        )

    # Utils

    def load_k8s_context(self):
        return self._load_k8s_context(
            cluster_id=self.cluster_id,
            resource_group=self.resource_group
        )

    @staticmethod
    def _load_k8s_context(cluster_id: int, resource_group: str):
        AzureController.load_aks_context(
            resource_group=resource_group,
            aks_name=f"{cluster_id}-aks"
        )
        config.load_kube_config(context=f"{cluster_id}-aks")


class ArmTemplateParameterBuilder:
    @staticmethod
    def create_aks_cluster(cluster_details: dict, export_path: str) -> dict:
        # Get params
        cluster_id = cluster_details['id']

        with open(f"{GlobalPaths.ABS_MARO_K8S_LIB}/clouds/aks/create_aks_cluster/parameters.json", "r") as f:
            base_parameters = json.load(f)
            parameters = base_parameters["parameters"]
            parameters["location"]["value"] = cluster_details["cloud"]["location"]
            parameters["adminUsername"]["value"] = cluster_details["user"]["admin_username"]
            parameters["adminPublicKey"]["value"] = cluster_details["user"]["admin_public_key"]
            parameters["clusterName"]["value"] = f"{cluster_id}-aks"
            parameters["agentCount"]["value"] = 1
            parameters["agentVMSize"]["value"] = cluster_details["master"]["node_size"]
            parameters["virtualNetworkName"]["value"] = f"{cluster_id}-vnet"
            parameters["acrName"]["value"] = f"{cluster_id}acr"
            parameters["acrSku"]["value"] = "Basic"
            parameters["storageAccountName"]["value"] = f"{cluster_id}st"
            parameters["fileShareName"]["value"] = f"{cluster_id}-fs"

        # Export parameters if the path is set
        if export_path:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w") as fw:
                json.dump(base_parameters, fw, indent=4)

        return parameters
