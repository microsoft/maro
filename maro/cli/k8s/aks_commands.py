# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64
import json
import os
import shutil
from os.path import abspath, dirname, expanduser, join

import yaml

# from maro.cli.k8s.executors.k8s_executor import K8sExecutor
# from maro.cli.utils.azure.acr import list_acr_repositories, login_acr
from maro.cli.utils import docker as docker_utils
from maro.cli.utils.azure import storage as azure_storage_utils
from maro.cli.utils.azure.aks import attach_acr
from maro.cli.utils.azure.deployment import create_deployment
from maro.cli.utils.azure.general import connect_to_aks, get_acr_push_permissions, set_env_credentials
from maro.cli.utils.azure.resource_group import create_resource_group, delete_resource_group_under_subscription
# from maro.cli.utils.azure.vm import list_vm_sizes
from maro.cli.utils.common import show_log
# from maro.cli.utils.deployment_validator import DeploymentValidator
# from maro.cli.utils.details_reader import DetailsReader
# from maro.cli.utils.details_writer import DetailsWriter
# from maro.cli.utils.name_creator import NameCreator
# from maro.cli.utils.path_convertor import PathConvertor
# from maro.cli.utils.subprocess import Subprocess
# from maro.utils.exception.cli_exception import BadRequestError, FileOperationError
from maro.rl.workflows.config import ConfigParser
from maro.utils.logger import CliLogger
from maro.utils.utils import LOCAL_MARO_ROOT

from .utils import k8s, k8s_manifest_generator

# metadata
CLI_K8S_PATH = dirname(abspath(__file__))
TEMPLATE_PATH = join(CLI_K8S_PATH, "test_template.json")
# TEMPLATE_PATH = join(CLI_K8S_PATH, "lib", "modes", "aks", "create_aks_cluster", "template.json")
NVIDIA_PLUGIN_PATH = join(CLI_K8S_PATH, "create_nvidia_plugin", "nvidia-device-plugin.yml")
LOCAL_ROOT = expanduser("~/.maro/aks")
DEPLOYMENT_CONF_PATH = os.path.join(LOCAL_ROOT, "conf.json")
DOCKER_FILE_PATH = join(LOCAL_MARO_ROOT, "docker_files", "dev.df")
DOCKER_IMAGE_NAME = "maro-aks"
REDIS_HOST = "maro-redis"
REDIS_PORT = 6379
ADDRESS_REGISTRY_NAME = "address-registry"
ADDRESS_REGISTRY_PORT = 6379
K8S_SECRET_NAME = "azure-secret"

# display
NO_DEPLOYMENT_MSG = "No Kubernetes deployment on Azure found. Use 'maro aks init' to create a deployment first"
NO_JOB_MSG = "No job named {} has been scheduled. Use 'maro aks job add' to add the job first."
JOB_EXISTS_MSG = "A job named {} has already been scheduled."

logger = CliLogger(name=__name__)


# helper functions
def get_resource_group_name(deployment_name: str):
    return f"rg-{deployment_name}"


def get_acr_name(deployment_name: str):
    return f"crmaro{deployment_name}"


def get_acr_server_name(acr_name: str):
    return f"{acr_name}.azurecr.io"


def get_docker_image_name_in_acr(acr_name: str, docker_image_name: str):
    return f"{get_acr_server_name(acr_name)}/{docker_image_name}"


def get_aks_name(deployment_name: str):
    return f"aks-maro-{deployment_name}"


def get_agentpool_name(deployment_name: str):
    return f"ap{deployment_name}"


def get_fileshare_name(deployment_name):
    return f"fs-{deployment_name}"


def get_storage_account_name(deployment_name: str):
    return f"stscenario{deployment_name}"


def get_virtual_network_name(location: str, deployment_name: str):
    return f"vnet-prod-{location}-{deployment_name}"


def get_local_job_path(job_name):
    return os.path.join(LOCAL_ROOT, job_name)


def get_storage_account_secret(resource_group_name: str, storage_account_name: str, namespace: str):
    storage_account_keys = azure_storage_utils.get_storage_account_keys(resource_group_name, storage_account_name)
    storage_key = storage_account_keys[0]["value"]
    secret_data = {
        "azurestorageaccountname": base64.b64encode(storage_account_name.encode()).decode(),
        "azurestorageaccountkey": base64.b64encode(bytes(storage_key.encode())).decode()
    }
    k8s.create_secret(K8S_SECRET_NAME, secret_data, namespace)


def get_resource_params(deployment_conf: dict) -> dict:
    """Create ARM parameters for Azure resource deployment ().

    See https://docs.microsoft.com/en-us/azure/azure-resource-manager/templates/overview for details.

    Args:
        deployment_conf (dict): Configuration dict for deployment on Azure.

    Returns:
        dict: parameter dict, should be exported to json.
    """
    name = deployment_conf["name"]
    return {
        "acrName": get_acr_name(name),
        "acrSku": deployment_conf["container_registry_service_tier"],
        "systemPoolVMCount": deployment_conf["hardware"]["k8s"]["vm_count"],
        "systemPoolVMSize": deployment_conf["hardware"]["k8s"]["vm_size"],
        "userPoolName": get_agentpool_name(name),
        "userPoolVMCount": deployment_conf["hardware"]["app"]["vm_count"],
        "userPoolVMSize": deployment_conf["hardware"]["app"]["vm_size"],
        "aksName": get_aks_name(name),
        "location": deployment_conf["location"],
        "storageAccountName": get_storage_account_name(name),
        "fileShareName": get_fileshare_name(name)
        # "virtualNetworkName": get_virtual_network_name(deployment_conf["location"], name)
    }


def prepare_docker_image_and_push_to_acr(image_name: str, context: str, docker_file_path: str, acr_name: str):
    # build and tag docker image locally and push to the Azure Container Registry
    if not docker_utils.image_exists(image_name):
        docker_utils.build_image(context, docker_file_path, image_name)

    get_acr_push_permissions(os.environ["AZURE_CLIENT_ID"], acr_name)
    docker_utils.push(image_name, get_acr_server_name(acr_name))


def start_redis_service_in_aks(host: str, port: int, namespace: str):
    k8s.load_config()
    k8s.create_namespace(namespace)
    k8s.create_deployment(k8s_manifest_generator.get_redis_deployment_manifest(host, port), namespace)
    k8s.create_service(k8s_manifest_generator.get_redis_service_manifest(host, port), namespace)


# CLI command functions
def init(deployment_conf_path: str, **kwargs):
    """Create MARO Cluster with create_deployment.

    Args:
        deployment_conf_path (str): Path to the deployment configuration file.
    """
    with open(deployment_conf_path, "r") as fp:
        deployment_conf = yaml.safe_load(fp)

    subscription = deployment_conf["azure_subscription"]
    name = deployment_conf["name"]
    if os.path.isfile(DEPLOYMENT_CONF_PATH):
        logger.warning(f"Deployment {name} has already been created")
        return

    os.makedirs(LOCAL_ROOT, exist_ok=True)
    resource_group_name = get_resource_group_name(name)
    try:
        set_env_credentials(LOCAL_ROOT, f"sp-{name}")

        # create resource group
        resource_group = create_resource_group(subscription, resource_group_name, deployment_conf["location"])
        logger.info_green(f"Provisioned resource group {resource_group.name} in {resource_group.location}")

        # Create ARM parameters and start deployment
        logger.info("Creating Azure resources...")
        resource_params = get_resource_params(deployment_conf)
        with open(TEMPLATE_PATH, 'r') as fp:
            template = json.load(fp)

        create_deployment(subscription, resource_group_name, name, template, resource_params)

        # Attach ACR to AKS
        aks_name, acr_name = resource_params["aksName"], resource_params["acrName"]
        attach_acr(resource_group_name, aks_name, acr_name)
        connect_to_aks(resource_group_name, aks_name)

        # build and tag docker image locally and push to the Azure Container Registry
        logger.info("Preparing docker image...")
        prepare_docker_image_and_push_to_acr(DOCKER_IMAGE_NAME, LOCAL_MARO_ROOT, DOCKER_FILE_PATH, acr_name)

        # start the Redis service in the k8s cluster in the deployment namespace and expose it
        logger.info("Starting Redis service in the k8s cluster...")
        start_redis_service_in_aks(REDIS_HOST, REDIS_PORT, name)

        # Dump the deployment configuration
        with open(DEPLOYMENT_CONF_PATH, "w") as fp:
            json.dump({
                "name": name,
                "subscription": subscription,
                "resource_group": resource_group_name,
                "resources": resource_params
            }, fp)
        logger.info_green(f"Cluster '{name}' is created")
    except Exception as e:
        # If failed, remove details folder, then raise
        shutil.rmtree(LOCAL_ROOT)
        logger.error_red(f"Deployment {name} failed due to {e}, rolling back...")
        delete_resource_group_under_subscription(subscription, resource_group_name)
    except KeyboardInterrupt:
        shutil.rmtree(LOCAL_ROOT)
        logger.error_red(f"Deployment {name} aborted, rolling back...")
        delete_resource_group_under_subscription(subscription, resource_group_name)


def add_job(conf_path: dict, **kwargs):
    if not os.path.isfile(DEPLOYMENT_CONF_PATH):
        logger.error_red(NO_DEPLOYMENT_MSG)
        return

    parser = ConfigParser(conf_path)
    job_conf = parser.config

    job_name = job_conf["job"]
    local_job_path = get_local_job_path(job_name)
    if os.path.isdir(local_job_path):
        logger.error_red(JOB_EXISTS_MSG.format(job_name))
        return

    os.makedirs(local_job_path)
    with open(DEPLOYMENT_CONF_PATH, "r") as fp:
        deployment_conf = json.load(fp)

    resource_group_name = deployment_conf["resource_group"]
    resource_name = deployment_conf["resources"]
    fileshare = azure_storage_utils.get_fileshare(resource_name["storageAccountName"], resource_name["fileShareName"])
    job_dir = azure_storage_utils.get_directory(fileshare, job_name)
    job_path_in_share = f"{resource_name['fileShareName']}/{job_name}"
    scenario_path = job_conf['scenario_path']
    logger.info(f"Uploading local directory {scenario_path}...")
    azure_storage_utils.upload_to_fileshare(job_dir, scenario_path, name="scenario")
    azure_storage_utils.get_directory(job_dir, "checkpoints")
    azure_storage_utils.get_directory(job_dir, "logs")

    # Define mount volumes, i.e., scenario code, checkpoints, logs and load point
    volumes = [
        k8s_manifest_generator.get_azurefile_volume_spec(name, f"{job_path_in_share}/{name}", K8S_SECRET_NAME)
        for name in ["scenario", "logs", "checkpoints"]
    ]

    if "load_path" in job_conf["training"]:
        load_dir = job_conf["training"]["load_path"]
        logger.info(f"Uploading local directory {load_dir}...")
        azure_storage_utils.upload_to_fileshare(job_dir, load_dir, name="loadpoint")
        volumes.append(
            k8s_manifest_generator.get_azurefile_volume_spec(
                "loadpoint", f"{job_path_in_share}/loadpoint", K8S_SECRET_NAME)
        )

    # Start k8s jobs
    k8s.load_config()
    k8s.create_namespace(job_name)
    get_storage_account_secret(resource_group_name, resource_name["storageAccountName"], job_name)
    k8s.create_service(
        k8s_manifest_generator.get_cross_namespace_service_access_manifest(
            ADDRESS_REGISTRY_NAME, REDIS_HOST, deployment_conf["name"], ADDRESS_REGISTRY_PORT
        ), job_name
    )
    for component_name, env in parser.as_env(containerize=True).items():
        container_spec = k8s_manifest_generator.get_container_spec(
            get_docker_image_name_in_acr(resource_name["acrName"], DOCKER_IMAGE_NAME),
            component_name,
            env,
            volumes
        )
        manifest = k8s_manifest_generator.get_job_manifest(
            resource_name["userPoolName"],
            component_name,
            container_spec,
            volumes
        )
        k8s.create_job(manifest, job_name)


def remove_jobs(job_names: str, **kwargs):
    if not os.path.isfile(DEPLOYMENT_CONF_PATH):
        logger.error_red(NO_DEPLOYMENT_MSG)
        return

    k8s.load_config()
    for job_name in job_names:
        local_job_path = get_local_job_path(job_name)
        if not os.path.isdir(local_job_path):
            logger.error_red(NO_JOB_MSG.format(job_name))
            return

        k8s.delete_job(job_name)


def get_job_logs(job_name: str, tail: int = -1, **kwargs):
    with open(DEPLOYMENT_CONF_PATH, "r") as fp:
        deployment_conf = json.load(fp)

    local_log_path = os.path.join(get_local_job_path(job_name), "log")
    resource_name = deployment_conf["resources"]
    fileshare = azure_storage_utils.get_fileshare(resource_name["storageAccountName"], resource_name["fileShareName"])
    job_dir = azure_storage_utils.get_directory(fileshare, job_name)
    log_dir = azure_storage_utils.get_directory(job_dir, "logs")
    azure_storage_utils.download_from_fileshare(log_dir, f"{job_name}.log", local_log_path)
    show_log(local_log_path, tail=tail)


def describe_job(job_name: str):
    pass

# def get_checkpoints(job_name: str, **kwargs):
#     with open(DEPLOYMENT_CONF_PATH, "r") as fp:
#         deployment_conf = json.load(fp)
#     local_checkpoint_path = job_conf.get("checkpoint_path", os.path.join(get_local_job_path, "ckpt"))
#     resource_name = deployment_conf["resources"]
#     fileshare = azure_storage_utils.get_fileshare(resource_name["storageAccountName"], resource_name["fileShareName"])
#     job_dir = azure_storage_utils.get_directory(fileshare, job_name)
#     azure_storage_utils.download_from_fileshare(job_dir, f"{job_name}.log", local_checkpoint_path)


def exit(**kwargs):
    try:
        with open(DEPLOYMENT_CONF_PATH, "r") as fp:
            deployment_conf = json.load(fp)
    except FileNotFoundError:
        logger.error(NO_DEPLOYMENT_MSG)
        return

    name = deployment_conf["name"]
    set_env_credentials(LOCAL_ROOT, f"sp-{name}")
    delete_resource_group_under_subscription(deployment_conf["subscription"], deployment_conf["resource_group"])


# class K8sAksExecutor(K8sExecutor):
#     """Executor for k8s/aks mode.

#     See https://maro.readthedocs.io/en/latest/key_components/orchestration.html for reference.
#     """

#     def __init__(self, cluster_name: str):
#         self.deployment_conf = DetailsReader.load_deployment_conf(cluster_name=cluster_name)

#         # Cloud configs
#         self.subscription = self.deployment_conf["cloud"]["subscription"]
#         self.resource_group = self.deployment_conf["cloud"]["resource_group"]
#         self.location = self.deployment_conf["cloud"]["location"]

#         super().__init__(deployment_conf=self.deployment_conf)

#     # maro k8s node
#     def scale_node(self, replicas: int, node_size: str) -> None:
#         """Scale up/down MARO Node.

#         Args:
#             replicas (int): desired number of MARO Node in specific node_size.
#             node_size (str): size of the MARO Node VM, see
#                 https://docs.microsoft.com/en-us/azure/virtual-machines/sizes for reference.
#
#         Returns:
#             None.
#         """
#         # Get node_size_to_info
#         node_size_to_info = self._get_node_size_to_info()

#         # Get node_size_to_spec, and check if node_size is valid
#         node_size_to_spec = self._get_node_size_to_spec()
#         if node_size not in node_size_to_spec:
#             raise BadRequestError(f"Invalid node_size '{node_size}'")

#         # Scale node
#         if node_size not in node_size_to_info:
#             self._build_node_pool(
#                 replicas=replicas,
#                 node_size=node_size
#             )
#         elif node_size_to_info[node_size]["count"] != replicas:
#             self._scale_node_pool(
#                 replicas=replicas,
#                 node_size=node_size,
#                 node_size_to_info=node_size_to_info
#             )
#         else:
#             logger.warning_yellow("Replica is match, no create or delete")

#     def _get_node_size_to_info(self) -> dict:
#         """Get node_size to info mapping of the K8s Cluster.

#         Returns:
#             dict: node_size to info mapping.
#         """
#         # List nodepool
#         nodepools = list_nodepool(
#             resource_group=self.resource_group,
#             aks_name=f"{self.cluster_id}-aks"
#         )

#         # Build node_size_to_count
#         node_size_to_count = {}
#         for nodepool in nodepools:
#             node_size_to_count[nodepool["vmSize"]] = nodepool

#         return node_size_to_count

#     def _get_node_size_to_spec(self) -> dict:
#         """Get node_size to spec mapping of Azure VM.

#         Returns:
#             dict: node_size to spec mapping.
#         """
#         # List available sizes for VM
#         specs = list_vm_sizes(location=self.location)

#         # Build node_size_to_spec
#         node_size_to_spec = {}
#         for spec in specs:
#             node_size_to_spec[spec["name"]] = spec

#         return node_size_to_spec

#     def _build_node_pool(self, replicas: int, node_size: str) -> None:
#         """Build node pool for the specific node_size.

#         Args:
#             replicas (int): number of MARO Node in specific node_size to stop.
#             node_size (str): size of the MARO Node VM,
#                 see https://docs.microsoft.com/en-us/azure/virtual-machines/sizes for reference.

#         Returns:
#             None.
#         """
#         logger.info(f"Building '{node_size}' nodepool")

#         # Build nodepool
#         add_nodepool(
#             resource_group=self.resource_group,
#             aks_name=f"{self.cluster_id}-aks",
#             nodepool_name=K8sAksExecutor._generate_nodepool_name(node_size=node_size),
#             node_count=replicas,
#             node_size=node_size
#         )

#         logger.info_green(f"'{node_size}' nodepool is built")

#     def _scale_node_pool(self, replicas: int, node_size: str, node_size_to_info: dict):
#         """Scale node pool of the specific node_size.

#         Args:
#             replicas (int): number of MARO Node in specific node_size to stop.
#             node_size (str): size of the MARO Node VM,
#                 see https://docs.microsoft.com/en-us/azure/virtual-machines/sizes for reference.
#             node_size_to_info (dict): node_size to info mapping.

#         Returns:
#             None.
#         """
#         logger.info(f"Scaling '{node_size}' nodepool")

#         # Scale node pool
#         scale_nodepool(
#             resource_group=self.resource_group,
#             aks_name=f"{self.cluster_id}-aks",
#             nodepool_name=node_size_to_info[node_size]["name"],
#             node_count=replicas
#         )

#         logger.info_green(f"'{node_size}' nodepool is scaled")

#     @staticmethod
#     def _generate_nodepool_name(node_size: str) -> str:
#         """Generate name of the nodepool.

#         Args:
#             node_size (str): size of the MARO Node VM.

#         Returns:
#             None.
#         """
#         return NameCreator.create_name_with_md5(prefix="pool", key=node_size, md5_len=8)

#     def list_node(self) -> None:
#         """Print node details to the command line.

#         Returns:
#             None.
#         """
#         # Get aks details
#         aks_details = get_aks(resource_group=self.resource_group, aks_name=f"{self.cluster_id}-aks")
#         agent_pools_details = aks_details["agentPoolProfiles"]

#         # Filter and print
#         node_details = {}
#         for agent_pool_details in agent_pools_details:
#             node_details[agent_pool_details["vmSize"]] = agent_pool_details["count"]
#         logger.info(
#             json.dumps(
#                 node_details,
#                 indent=4, sort_keys=True
#             )
#         )

#     # maro k8s image

#     def push_image(self, image_name: str) -> None:
#         """Push local image to the MARO Cluster.

#         Args:
#             image_name (str): name of the local image that loaded in the docker.

#         Returns:
#             None.
#         """
#         remote_image_name = f"{self.cluster_id}acr.azurecr.io/{image_name}"

#         # ACR login
#         login_acr(acr_name=f"{self.cluster_id}acr")

#         # Tag image
#         command = f"docker tag {image_name} {remote_image_name}"
#         _ = Subprocess.run(command=command)

#         # Push image to ACR
#         command = f"docker push {remote_image_name}"
#         _ = Subprocess.run(command=command)

#     def list_image(self):
#         """Print image details to the command line.

#         Returns:
#             None.
#         """
#         # List acr repository
#         acr_repositories = list_acr_repositories(acr_name=f"{self.cluster_id}acr")
#         logger.info(acr_repositories)

#     # maro k8s data

#     def push_data(self, local_path: str, remote_dir: str) -> None:
#         """Push local data to the remote AFS service via azcopy.

#         Args:
#             local_path (str): path of the local data.
#             remote_dir (str): path of the remote folder.

#         Returns:
#             None.
#         """
#         # Get sas
#         sas = self._check_and_get_account_sas()

#         # Push data
#         abs_local_path = os.path.expanduser(local_path)
#         abs_source_path = PathConvertor.build_path_without_trailing_slash(abs_local_path)
#         target_dir = PathConvertor.build_path_with_trailing_slash(remote_dir)
#         if not target_dir.startswith("/"):
#             raise FileOperationError(f"Invalid remote path: {target_dir}\nShould be started with '/'")
#         copy_command = (
#             "azcopy copy "
#             f"'{abs_source_path}' "
#             f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs{target_dir}?{sas}' "
#             "--recursive=True"
#         )
#         _ = Subprocess.run(command=copy_command)

#     def pull_data(self, local_dir: str, remote_path: str) -> None:
#         """Pull remote AFS service data to local folder via azcopy.

#         Args:
#             local_dir (str): path of the local folder.
#             remote_path (str): path of the remote data.

#         Returns:
#             None.
#         """
#         # Get sas
#         sas = self._check_and_get_account_sas()

#         # Push data
#         abs_local_dir = os.path.expanduser(local_dir)
#         source_path = PathConvertor.build_path_without_trailing_slash(remote_path)
#         abs_target_dir = PathConvertor.build_path_with_trailing_slash(abs_local_dir)
#         os.makedirs(abs_target_dir, exist_ok=True)
#         if not source_path.startswith("/"):
#             raise FileOperationError(f"Invalid remote path: {source_path}\nShould be started with '/'")
#         copy_command = (
#             "azcopy copy "
#             f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs{source_path}?{sas}' "
#             f"'{abs_target_dir}' "
#             "--recursive=True"
#         )
#         _ = Subprocess.run(command=copy_command)

#     def remove_data(self, remote_path: str) -> None:
#         """Remote data at the remote AFS service.

#         Args:
#             remote_path (str): path of the remote data.

#         Returns:
#             None.
#         """
#         # FIXME: Remove failed, The specified resource may be in use by an SMB client

#         # Get sas
#         sas = self._check_and_get_account_sas()

#         # Remove data
#         copy_command = (
#             "azcopy remove "
#             f"'https://{self.cluster_id}st.file.core.windows.net/{self.cluster_id}-fs{remote_path}?{sas}' "
#             "--recursive=True"
#         )
#         _ = Subprocess.run(command=copy_command)

#     def _check_and_get_account_sas(self) -> str:
#         """Check and get account sas token, also update it to the deployment_conf.

#         Ref: https://msdn.microsoft.com/library/azure/mt584140.aspx

#         Returns:
#             str: account sas token.
#         """

#         # Load details
#         cloud_details = self.deployment_conf["cloud"]

#         # Regenerate sas if the key is None or expired TODO:
#         if "account_sas" not in cloud_details:
#             account_sas = get_storage_account_sas(account_name=f"{self.cluster_id}st")
#             cloud_details["account_sas"] = account_sas
#             DetailsWriter.save_deployment_conf(
#                 cluster_name=self.cluster_name,
#                 deployment_conf=self.deployment_conf
#             )

#         return cloud_details["account_sas"]

#     # maro k8s job

#     def _create_k8s_job(self, job_details: dict) -> dict:
#         """Create k8s job object with job_details.

#         Args:
#             job_details (dict): details of the MARO Job.

#         Returns:
#             dict: k8s job object.
#         """
#         # Get config template
#         with open(f"{MARO_K8S_LIB}/modes/aks/create_job/job.yml") as fr:
#             k8s_job_config = yaml.safe_load(fr)
#         with open(f"{MARO_K8S_LIB}/modes/aks/create_job/container.yml") as fr:
#             k8s_container_config = yaml.safe_load(fr)

#         # Fill configs
#         k8s_job_config["metadata"]["name"] = job_details["id"]
#         k8s_job_config["metadata"]["labels"]["jobName"] = job_details["name"]
#         azure_file_config = k8s_job_config["spec"]["template"]["spec"]["volumes"][0]["azureFile"]
#         azure_file_config["secretName"] = "azure-storage-account-secret"
#         azure_file_config["shareName"] = f"{self.cluster_id}-fs"

#         # Create and fill container config
#         for component_type, component_details in job_details["components"].items():
#             for component_index in range(component_details["num"]):
#                 container_config = self._create_k8s_container_config(
#                     job_details=job_details,
#                     k8s_container_config_template=k8s_container_config,
#                     component_type=component_type,
#                     component_index=component_index
#                 )
#                 k8s_job_config["spec"]["template"]["spec"]["containers"].append(container_config)

#         return k8s_job_config

#     def _create_k8s_container_config(
#         self, job_details: dict, k8s_container_config_template: dict,
#         component_type: str, component_index: int
#     ) -> dict:
#         """Create the container config in the k8s job object.

#         Args:
#             job_details (dict): details of the MARO Job.
#             k8s_container_config_template (dict): template of the k8s_container_config.
#             component_type (str): type of the component.
#             component_index (int): index of the component.

#         Returns:
#             dict: the container config.
#         """
#         # Copy config.
#         k8s_container_config = copy.deepcopy(k8s_container_config_template)

#         # Load details
#         component_details = job_details["components"][component_type]
#         job_id = job_details["id"]
#         component_id = job_details["components"][component_type]["id"]
#         container_name = f"{job_id}-{component_id}-{component_index}"

#         # Fill configs.
#         k8s_container_config["name"] = container_name
#         k8s_container_config["image"] = self._build_image_address(image_name=component_details["image"])
#         k8s_container_config["resources"]["requests"] = {
#             "cpu": component_details["resources"]["cpu"],
#             "memory": component_details["resources"]["memory"],
#             "nvidia.com/gpu": component_details["resources"]["gpu"]
#         }
#         k8s_container_config["resources"]["limits"] = {
#             "cpu": component_details["resources"]["cpu"],
#             "memory": component_details["resources"]["memory"],
#             "nvidia.com/gpu": component_details["resources"]["gpu"]
#         }
#         k8s_container_config["env"] = [
#             {
#                 "name": "CLUSTER_ID",
#                 "value": f"{self.cluster_id}"
#             },
#             {
#                 "name": "CLUSTER_NAME",
#                 "value": f"{self.cluster_name}"
#             },
#             {
#                 "name": "JOB_ID",
#                 "value": job_id
#             },
#             {
#                 "name": "JOB_NAME",
#                 "value": job_details["name"]
#             },
#             {
#                 "name": "COMPONENT_ID",
#                 "value": component_id
#             },
#             {
#                 "name": "COMPONENT_TYPE",
#                 "value": f"{component_type}"
#             },
#             {
#                 "name": "COMPONENT_INDEX",
#                 "value": f"{component_index}"
#             },
#             {
#                 "name": "PYTHONUNBUFFERED",
#                 "value": "0"
#             }
#         ]
#         k8s_container_config["command"] = component_details["command"]
#         k8s_container_config["volumeMounts"][0]["mountPath"] = component_details["mount"]["target"]

#         return k8s_container_config

#     def _build_image_address(self, image_name: str) -> str:
#         """Build image address name for image that stored at Azure Container Registry.

#         Args:
#             image_name (str): name of the image.

#         Returns:
#             str: image address name.
#         """
#         # Get repositories
#         acr_repositories = list_acr_repositories(acr_name=f"{self.cluster_id}acr")

#         # Build address
#         if image_name in acr_repositories:
#             return f"{self.cluster_id}acr.azurecr.io/{image_name}"
#         else:
#             return image_name

#     @staticmethod
#     def _export_log(pod_id: str, container_name: str, export_dir: str) -> None:
#         """Export k8s job logs to the specific folder.

#         Args:
#             pod_id (str): id of the k8s pod.
#             container_name (str): name of the container.
#             export_dir (str): path of the exported folder.

#         Returns:
#             None.
#         """
#         os.makedirs(os.path.expanduser(export_dir + f"/{pod_id}"), exist_ok=True)
#         with open(os.path.expanduser(export_dir + f"/{pod_id}/{container_name}.log"), "w") as fw:
#             return_str = client.CoreV1Api().read_namespaced_pod_log(name=pod_id, namespace="default")
#             fw.write(return_str)

#     # maro k8s status

#     def status(self) -> None:
#         """Print details of specific MARO Resources (redis only at this time).

#         Returns:
#             None.
#         """
#         return_status = {}

#         # Get pods details
#         pod_list = client.CoreV1Api().list_pod_for_all_namespaces(watch=False).to_dict()["items"]

#         for pod in pod_list:
#             if "app" in pod["metadata"]["labels"] and pod["metadata"]["labels"]["app"] == "maro-redis":
#                 return_status["redis"] = {
#                     "private_ip_address": pod["status"]["pod_ip"]
#                 }
#                 break

#         # Print status
#         logger.info(
#             json.dumps(
#                 return_status,
#                 indent=4, sort_keys=True
#             )
#         )

#     # Utils

#     def load_k8s_context(self) -> None:
#         """Activate load k8s context operation.

#         Returns:
#             None.
#         """
#         self._load_k8s_context(
#             cluster_id=self.cluster_id,
#             resource_group=self.resource_group
#         )

#     @staticmethod
#     def _load_k8s_context(cluster_id: int, resource_group: str) -> None:
#         """Load the k8s context.

#         Set current k8s context (only in the CLI runtime) to the k8s cluster that related to the MARO Cluster.

#         Args:
#             cluster_id (str): id of the MARO Cluster.
#             resource_group (str): name of the resource group.

#         Returns:
#             None.
#         """
#         load_aks_context(
#             resource_group=resource_group,
#             aks_name=f"{cluster_id}-aks"
#         )
#         config.load_kube_config(context=f"{cluster_id}-aks")
