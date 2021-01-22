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
from maro.cli.k8s.utils.params import K8sPaths
from maro.cli.utils.azure_controller import AzureController
from maro.cli.utils.deployment_validator import DeploymentValidator
from maro.cli.utils.details_reader import DetailsReader
from maro.cli.utils.details_writer import DetailsWriter
from maro.cli.utils.name_creator import NameCreator
from maro.cli.utils.params import GlobalParams, GlobalPaths
from maro.cli.utils.path_convertor import PathConvertor
from maro.cli.utils.subprocess import Subprocess
from maro.utils.exception.cli_exception import BadRequestError, FileOperationError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class K8sAksExecutor(K8sExecutor):
    """Executor for k8s/aks mode.

    See https://maro.readthedocs.io/en/latest/key_components/orchestration.html for reference.
    """

    def __init__(self, cluster_name: str):
        self.cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

        # Cloud configs
        self.subscription = self.cluster_details["cloud"]["subscription"]
        self.resource_group = self.cluster_details["cloud"]["resource_group"]
        self.location = self.cluster_details["cloud"]["location"]

        super().__init__(cluster_details=self.cluster_details)

    # maro k8s create

    @staticmethod
    def create(create_deployment: dict) -> None:
        """Create MARO Cluster with create_deployment.

        Args:
            create_deployment (dict): create_deployment of k8s/aks. See lib/deployments/internal for reference.

        Returns:
            None.
        """
        logger.info("Creating cluster")

        # Get standardized cluster_details
        cluster_details = K8sAksExecutor._standardize_cluster_details(create_deployment=create_deployment)
        cluster_name = cluster_details["name"]
        if os.path.isdir(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}"):
            raise BadRequestError(f"Cluster '{cluster_name}' is exist")

        # Start creating
        try:
            K8sAksExecutor._create_resource_group(cluster_details=cluster_details)
            K8sAksExecutor._create_k8s_cluster(cluster_details=cluster_details)
            K8sAksExecutor._load_k8s_context(
                cluster_id=cluster_details["id"],
                resource_group=cluster_details["cloud"]["resource_group"]
            )
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
    def _standardize_cluster_details(create_deployment: dict) -> dict:
        """Standardize cluster_details from create_deployment.

        We use create_deployment to build cluster_details (they share the same keys structure).

        Args:
            create_deployment (dict): create_deployment of k8s/aks. See lib/deployments/internal for reference.

        Returns:
            dict: standardized cluster_details.
        """
        optional_key_to_value = {
            "root['master']['redis']": {
                "port": GlobalParams.DEFAULT_REDIS_PORT
            },
            "root['master']['redis']['port']": GlobalParams.DEFAULT_REDIS_PORT
        }
        with open(f"{K8sPaths.ABS_MARO_K8S_LIB}/deployments/internal/k8s_aks_create.yml") as fr:
            create_deployment_template = yaml.safe_load(fr)
        DeploymentValidator.validate_and_fill_dict(
            template_dict=create_deployment_template,
            actual_dict=create_deployment,
            optional_key_to_value=optional_key_to_value
        )

        # Init runtime fields
        create_deployment["id"] = NameCreator.create_cluster_id()

        return create_deployment

    @staticmethod
    def _create_resource_group(cluster_details: dict) -> None:
        """Create the resource group if it does not exist.

        Args:
            cluster_details (dict): details of the cluster.

        Returns:
            None.
        """

        # Get params
        subscription = cluster_details["cloud"]["subscription"]
        resource_group = cluster_details["cloud"]["resource_group"]

        # Check if Azure CLI is installed, and print version
        azure_version = AzureController.get_version()
        logger.info_green(f"Your Azure CLI version: {azure_version['azure-cli']}")

        # Set subscription id
        AzureController.set_subscription(subscription=subscription)
        logger.info_green(f"Set subscription to '{subscription}'")

        # Check and create resource group
        resource_group_info = AzureController.get_resource_group(resource_group=resource_group)
        if resource_group_info:
            logger.warning_yellow(f"Azure resource group '{resource_group}' already exists")
        else:
            AzureController.create_resource_group(
                resource_group=resource_group,
                location=cluster_details["cloud"]["location"]
            )
            logger.info_green(f"Resource group '{resource_group}' is created")

    @staticmethod
    def _create_k8s_cluster(cluster_details: dict) -> None:
        """Create k8s cluster for the MARO Cluster.

        Args:
            cluster_details (dict): details of the MARO Cluster.

        Returns:
            None.
        """
        logger.info("Creating k8s cluster")

        # Create ARM parameters and start deployment
        template_file_path = f"{K8sPaths.ABS_MARO_K8S_LIB}/modes/aks/create_aks_cluster/template.json"
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
        AzureController.attach_acr(
            resource_group=cluster_details["cloud"]["resource_group"],
            aks_name=f"{cluster_details['id']}-aks",
            acr_name=f"{cluster_details['id']}acr"
        )

        logger.info_green("K8s cluster is created")

    @staticmethod
    def _init_nvidia_plugin() -> None:
        """Setup nvidia plugin for the MARO Cluster.

        Returns:
            None.
        """
        client.CoreV1Api().create_namespace(body=client.V1Namespace(metadata=client.V1ObjectMeta(name="gpu-resources")))

        with open(
            f"{K8sPaths.ABS_MARO_K8S_LIB}/modes/aks/create_nvidia_plugin/nvidia-device-plugin.yml", "r"
        ) as fr:
            redis_deployment = yaml.safe_load(fr)
        client.AppsV1Api().create_namespaced_daemon_set(body=redis_deployment, namespace="gpu-resources")

    @staticmethod
    def _create_storage_account_secret(cluster_details: dict) -> None:
        """Setup storage_account_secret for the MARO Cluster.

        The secret is used in Azure File Service.

        Returns:
            None.
        """
        # Build params
        storage_account_name = f"{cluster_details['id']}st"

        # Get storage account key
        storage_account_keys = AzureController.get_storage_account_keys(
            resource_group=cluster_details["cloud"]["resource_group"],
            storage_account_name=storage_account_name
        )
        storage_key = storage_account_keys[0]["value"]

        # Create k8s secret
        client.CoreV1Api().create_namespaced_secret(
            body=client.V1Secret(
                metadata=client.V1ObjectMeta(name="azure-storage-account-secret"),
                data={
                    "azurestorageaccountname": base64.b64encode(storage_account_name.encode()).decode(),
                    "azurestorageaccountkey": base64.b64encode(bytes(storage_key.encode())).decode()
                }
            ),
            namespace="default"
        )

    # maro k8s delete

    def delete(self) -> None:
        """Delete the MARO Cluster.

        Returns:
            None.
        """
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
            AzureController.delete_resources(resource_ids=deletable_ids)

        # Delete cluster folder
        shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}")

        logger.info_green(f"Cluster '{self.cluster_name}' is deleted")

    # maro k8s node

    def scale_node(self, replicas: int, node_size: str) -> None:
        """Scale up/down MARO Node.

        Args:
            replicas (int): desired number of MARO Node in specific node_size.
            node_size (str): size of the MARO Node VM, see https://docs.microsoft.com/en-us/azure/virtual-machines/sizes
                for reference.

        Returns:
            None.
        """
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

    def _get_node_size_to_info(self) -> dict:
        """Get node_size to info mapping of the K8s Cluster.

        Returns:
            dict: node_size to info mapping.
        """
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
        """Get node_size to spec mapping of Azure VM.

        Returns:
            dict: node_size to spec mapping.
        """
        # List available sizes for VM
        specs = AzureController.list_vm_sizes(location=self.location)

        # Build node_size_to_spec
        node_size_to_spec = {}
        for spec in specs:
            node_size_to_spec[spec["name"]] = spec

        return node_size_to_spec

    def _build_node_pool(self, replicas: int, node_size: str) -> None:
        """Build node pool for the specific node_size.

        Args:
            replicas (int): number of MARO Node in specific node_size to stop.
            node_size (str): size of the MARO Node VM,
                see https://docs.microsoft.com/en-us/azure/virtual-machines/sizes for reference.

        Returns:
            None.
        """
        logger.info(f"Building '{node_size}' nodepool")

        # Build nodepool
        AzureController.add_nodepool(
            resource_group=self.resource_group,
            aks_name=f"{self.cluster_id}-aks",
            nodepool_name=K8sAksExecutor._generate_nodepool_name(node_size=node_size),
            node_count=replicas,
            node_size=node_size
        )

        logger.info_green(f"'{node_size}' nodepool is built")

    def _scale_node_pool(self, replicas: int, node_size: str, node_size_to_info: dict):
        """Scale node pool of the specific node_size.

        Args:
            replicas (int): number of MARO Node in specific node_size to stop.
            node_size (str): size of the MARO Node VM,
                see https://docs.microsoft.com/en-us/azure/virtual-machines/sizes for reference.
            node_size_to_info (dict): node_size to info mapping.

        Returns:
            None.
        """
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
    def _generate_nodepool_name(node_size: str) -> str:
        """Generate name of the nodepool.

        Args:
            node_size (str): size of the MARO Node VM.

        Returns:
            None.
        """
        return NameCreator.create_name_with_md5(prefix="pool", key=node_size, md5_len=8)

    def list_node(self) -> None:
        """Print node details to the command line.

        Returns:
            None.
        """
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

    def push_image(self, image_name: str) -> None:
        """Push local image to the MARO Cluster.

        Args:
            image_name (str): name of the local image that loaded in the docker.

        Returns:
            None.
        """
        remote_image_name = f"{self.cluster_id}acr.azurecr.io/{image_name}"

        # ACR login
        AzureController.login_acr(acr_name=f"{self.cluster_id}acr")

        # Tag image
        command = f"docker tag {image_name} {remote_image_name}"
        _ = Subprocess.run(command=command)

        # Push image to ACR
        command = f"docker push {remote_image_name}"
        _ = Subprocess.run(command=command)

    def list_image(self):
        """Print image details to the command line.

        Returns:
            None.
        """
        # List acr repository
        acr_repositories = AzureController.list_acr_repositories(acr_name=f"{self.cluster_id}acr")
        logger.info(acr_repositories)

    # maro k8s data

    def push_data(self, local_path: str, remote_dir: str) -> None:
        """Push local data to the remote AFS service via azcopy.

        Args:
            local_path (str): path of the local data.
            remote_dir (str): path of the remote folder.

        Returns:
            None.
        """
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
        _ = Subprocess.run(command=copy_command)

    def pull_data(self, local_dir: str, remote_path: str) -> None:
        """Pull remote AFS service data to local folder via azcopy.

        Args:
            local_dir (str): path of the local folder.
            remote_path (str): path of the remote data.

        Returns:
            None.
        """
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
        _ = Subprocess.run(command=copy_command)

    def remove_data(self, remote_path: str) -> None:
        """Remote data at the remote AFS service.

        Args:
            remote_path (str): path of the remote data.

        Returns:
            None.
        """
        # FIXME: Remove failed, The specified resource may be in use by an SMB client

        # Get sas
        sas = self._check_and_get_account_sas()

        # Remove data
        copy_command = (
            "azcopy remove "
            f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs{remote_path}?{sas}' "
            "--recursive=True"
        )
        _ = Subprocess.run(command=copy_command)

    def _check_and_get_account_sas(self) -> str:
        """Check and get account sas token, also update it to the cluster_details.

        Ref: https://msdn.microsoft.com/library/azure/mt584140.aspx

        Returns:
            str: account sas token.
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
        """Create k8s job object with job_details.

        Args:
            job_details (dict): details of the MARO Job.

        Returns:
            dict: k8s job object.
        """
        # Get config template
        with open(f"{K8sPaths.ABS_MARO_K8S_LIB}/modes/aks/create_job/job.yml") as fr:
            k8s_job_config = yaml.safe_load(fr)
        with open(f"{K8sPaths.ABS_MARO_K8S_LIB}/modes/aks/create_job/container.yml") as fr:
            k8s_container_config = yaml.safe_load(fr)

        # Fill configs
        k8s_job_config["metadata"]["name"] = job_details["id"]
        k8s_job_config["metadata"]["labels"]["jobName"] = job_details["name"]
        azure_file_config = k8s_job_config["spec"]["template"]["spec"]["volumes"][0]["azureFile"]
        azure_file_config["secretName"] = "azure-storage-account-secret"
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
    ) -> dict:
        """Create the container config in the k8s job object.

        Args:
            job_details (dict): details of the MARO Job.
            k8s_container_config_template (dict): template of the k8s_container_config.
            component_type (str): type of the component.
            component_index (int): index of the component.

        Returns:
            dict: the container config.
        """
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
                "value": job_id
            },
            {
                "name": "JOB_NAME",
                "value": job_details["name"]
            },
            {
                "name": "COMPONENT_ID",
                "value": component_id
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
        """Build image address name for image that stored at Azure Container Registry.

        Args:
            image_name (str): name of the image.

        Returns:
            str: image address name.
        """
        # Get repositories
        acr_repositories = AzureController.list_acr_repositories(acr_name=f"{self.cluster_id}acr")

        # Build address
        if image_name in acr_repositories:
            return f"{self.cluster_id}acr.azurecr.io/{image_name}"
        else:
            return image_name

    @staticmethod
    def _export_log(pod_id: str, container_name: str, export_dir: str) -> None:
        """Export k8s job logs to the specific folder.

        Args:
            pod_id (str): id of the k8s pod.
            container_name (str): name of the container.
            export_dir (str): path of the exported folder.

        Returns:
            None.
        """
        os.makedirs(os.path.expanduser(export_dir + f"/{pod_id}"), exist_ok=True)
        with open(os.path.expanduser(export_dir + f"/{pod_id}/{container_name}.log"), "w") as fw:
            return_str = client.CoreV1Api().read_namespaced_pod_log(name=pod_id, namespace="default")
            fw.write(return_str)

    # maro k8s status

    def status(self) -> None:
        """Print details of specific MARO Resources (redis only at this time).

        Returns:
            None.
        """
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

    def load_k8s_context(self) -> None:
        """Activate load k8s context operation.

        Returns:
            None.
        """
        self._load_k8s_context(
            cluster_id=self.cluster_id,
            resource_group=self.resource_group
        )

    @staticmethod
    def _load_k8s_context(cluster_id: int, resource_group: str) -> None:
        """Load the k8s context.

        Set current k8s context (only in the CLI runtime) to the k8s cluster that related to the MARO Cluster.

        Args:
            cluster_id (str): id of the MARO Cluster.
            resource_group (str): name of the resource group.

        Returns:
            None.
        """
        AzureController.load_aks_context(
            resource_group=resource_group,
            aks_name=f"{cluster_id}-aks"
        )
        config.load_kube_config(context=f"{cluster_id}-aks")


class ArmTemplateParameterBuilder:
    @staticmethod
    def create_aks_cluster(cluster_details: dict, export_path: str) -> dict:
        """Create parameters file for AKS cluster.

        Args:
            cluster_details (dict): details of the MARO Cluster.
            export_path (str): path to export the parameter file.

        Returns:
            dict: parameter dict, should be exported to json.
        """

        # Get params
        cluster_id = cluster_details['id']

        with open(f"{K8sPaths.ABS_MARO_K8S_LIB}/modes/aks/create_aks_cluster/parameters.json", "r") as f:
            base_parameters = json.load(f)
            parameters = base_parameters["parameters"]
            parameters["acrName"]["value"] = f"{cluster_id}acr"
            parameters["acrSku"]["value"] = "Basic"
            parameters["adminPublicKey"]["value"] = cluster_details["cloud"]["default_public_key"]
            parameters["adminUsername"]["value"] = cluster_details["cloud"]["default_username"]
            parameters["agentCount"]["value"] = 1
            parameters["agentVMSize"]["value"] = cluster_details["master"]["node_size"]
            parameters["clusterName"]["value"] = f"{cluster_id}-aks"
            parameters["fileShareName"]["value"] = f"{cluster_id}-fs"
            parameters["location"]["value"] = cluster_details["cloud"]["location"]
            parameters["storageAccountName"]["value"] = f"{cluster_id}st"
            parameters["virtualNetworkName"]["value"] = f"{cluster_id}-vnet"

        # Export parameters if the path is set
        if export_path:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w") as fw:
                json.dump(base_parameters, fw, indent=4)

        return parameters
