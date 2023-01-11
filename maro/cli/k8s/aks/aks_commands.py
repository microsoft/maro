# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64
import json
import os
import shutil
from os.path import abspath, dirname, expanduser, join

import yaml

from maro.cli.utils import docker as docker_utils
from maro.cli.utils.azure import storage as azure_storage_utils
from maro.cli.utils.azure.aks import attach_acr
from maro.cli.utils.azure.deployment import create_deployment
from maro.cli.utils.azure.general import connect_to_aks, get_acr_push_permissions, set_env_credentials
from maro.cli.utils.azure.resource_group import create_resource_group, delete_resource_group
from maro.cli.utils.common import show_log
from maro.rl.workflows.config import ConfigParser
from maro.utils.logger import CliLogger
from maro.utils.utils import LOCAL_MARO_ROOT

from ..utils import k8s_manifest_generator, k8s_ops

# metadata
CLI_AKS_PATH = dirname(abspath(__file__))
TEMPLATE_PATH = join(CLI_AKS_PATH, "template.json")
NVIDIA_PLUGIN_PATH = join(CLI_AKS_PATH, "create_nvidia_plugin", "nvidia-device-plugin.yml")
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


def get_fileshare_name(deployment_name: str):
    return f"fs-{deployment_name}"


def get_storage_account_name(deployment_name: str):
    return f"stscenario{deployment_name}"


def get_virtual_network_name(location: str, deployment_name: str):
    return f"vnet-prod-{location}-{deployment_name}"


def get_local_job_path(job_name: str):
    return os.path.join(LOCAL_ROOT, job_name)


def get_storage_account_secret(resource_group_name: str, storage_account_name: str, namespace: str):
    storage_account_keys = azure_storage_utils.get_storage_account_keys(resource_group_name, storage_account_name)
    storage_key = storage_account_keys[0]["value"]
    secret_data = {
        "azurestorageaccountname": base64.b64encode(storage_account_name.encode()).decode(),
        "azurestorageaccountkey": base64.b64encode(bytes(storage_key.encode())).decode(),
    }
    k8s_ops.create_secret(K8S_SECRET_NAME, secret_data, namespace)


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
        "systemPoolVMCount": deployment_conf["resources"]["k8s"]["vm_count"],
        "systemPoolVMSize": deployment_conf["resources"]["k8s"]["vm_size"],
        "userPoolName": get_agentpool_name(name),
        "userPoolVMCount": deployment_conf["resources"]["app"]["vm_count"],
        "userPoolVMSize": deployment_conf["resources"]["app"]["vm_size"],
        "aksName": get_aks_name(name),
        "location": deployment_conf["location"],
        "storageAccountName": get_storage_account_name(name),
        "fileShareName": get_fileshare_name(name),
        # "virtualNetworkName": get_virtual_network_name(deployment_conf["location"], name)
    }


def prepare_docker_image_and_push_to_acr(image_name: str, context: str, docker_file_path: str, acr_name: str):
    # build and tag docker image locally and push to the Azure Container Registry
    if not docker_utils.image_exists(image_name):
        docker_utils.build_image(context, docker_file_path, image_name)

    get_acr_push_permissions(os.environ["AZURE_CLIENT_ID"], acr_name)
    docker_utils.push(image_name, get_acr_server_name(acr_name))


def start_redis_service_in_aks(host: str, port: int, namespace: str):
    k8s_ops.load_config()
    k8s_ops.create_namespace(namespace)
    k8s_ops.create_deployment(k8s_manifest_generator.get_redis_deployment_manifest(host, port), namespace)
    k8s_ops.create_service(k8s_manifest_generator.get_redis_service_manifest(host, port), namespace)


# CLI command functions
def init(deployment_conf_path: str, **kwargs):
    """Prepare Azure resources needed for an AKS cluster using a YAML configuration file.

    The configuration file template can be found in cli/k8s/aks/conf.yml. Use the Azure CLI to log into
    your Azure account (az login ...) and the the Azure Container Registry (az acr login ...) first.

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
        # Set credentials as environment variables
        set_env_credentials(LOCAL_ROOT, f"sp-{name}")

        # create resource group
        resource_group = create_resource_group(subscription, resource_group_name, deployment_conf["location"])
        logger.info_green(f"Provisioned resource group {resource_group.name} in {resource_group.location}")

        # Create ARM parameters and start deployment
        logger.info("Creating Azure resources...")
        resource_params = get_resource_params(deployment_conf)
        with open(TEMPLATE_PATH, "r") as fp:
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
            json.dump(
                {
                    "name": name,
                    "subscription": subscription,
                    "resource_group": resource_group_name,
                    "resources": resource_params,
                },
                fp,
            )
        logger.info_green(f"Cluster '{name}' is created")
    except Exception as e:
        # If failed, remove details folder, then raise
        shutil.rmtree(LOCAL_ROOT)
        logger.error_red(f"Deployment {name} failed due to {e}, rolling back...")
        delete_resource_group(subscription, resource_group_name)
    except KeyboardInterrupt:
        shutil.rmtree(LOCAL_ROOT)
        logger.error_red(f"Deployment {name} aborted, rolling back...")
        delete_resource_group(subscription, resource_group_name)


def add_job(conf_path: dict, **kwargs):
    if not os.path.isfile(DEPLOYMENT_CONF_PATH):
        logger.error_red(NO_DEPLOYMENT_MSG)
        return

    parser = ConfigParser(conf_path)
    job_name = parser.config["job"]
    local_job_path = get_local_job_path(job_name)
    if os.path.isdir(local_job_path):
        logger.error_red(JOB_EXISTS_MSG.format(job_name))
        return

    os.makedirs(local_job_path)
    with open(DEPLOYMENT_CONF_PATH, "r") as fp:
        deployment_conf = json.load(fp)

    resource_group_name, resource_name = deployment_conf["resource_group"], deployment_conf["resources"]
    fileshare = azure_storage_utils.get_fileshare(resource_name["storageAccountName"], resource_name["fileShareName"])
    job_dir = azure_storage_utils.get_directory(fileshare, job_name)
    scenario_path = parser.config["scenario_path"]
    logger.info(f"Uploading local directory {scenario_path}...")
    azure_storage_utils.upload_to_fileshare(job_dir, scenario_path, name="scenario")
    azure_storage_utils.get_directory(job_dir, "checkpoints")
    azure_storage_utils.get_directory(job_dir, "logs")

    # Define mount volumes, i.e., scenario code, checkpoints, logs and load point
    job_path_in_share = f"{resource_name['fileShareName']}/{job_name}"
    volumes = [
        k8s_manifest_generator.get_azurefile_volume_spec(name, f"{job_path_in_share}/{name}", K8S_SECRET_NAME)
        for name in ["scenario", "logs", "checkpoints"]
    ]

    if "load_path" in parser.config["training"]:
        load_path = parser.config["training"]["load_path"]
        logger.info(f"Uploading local model directory {load_path}...")
        azure_storage_utils.upload_to_fileshare(job_dir, load_path, name="loadpoint")
        volumes.append(
            k8s_manifest_generator.get_azurefile_volume_spec(
                "loadpoint",
                f"{job_path_in_share}/loadpoint",
                K8S_SECRET_NAME,
            ),
        )

    # Start k8s jobs
    k8s_ops.load_config()
    k8s_ops.create_namespace(job_name)
    get_storage_account_secret(resource_group_name, resource_name["storageAccountName"], job_name)
    k8s_ops.create_service(
        k8s_manifest_generator.get_cross_namespace_service_access_manifest(
            ADDRESS_REGISTRY_NAME,
            REDIS_HOST,
            deployment_conf["name"],
            ADDRESS_REGISTRY_PORT,
        ),
        job_name,
    )
    for component_name, (script, env) in parser.get_job_spec(containerize=True).items():
        container_spec = k8s_manifest_generator.get_container_spec(
            get_docker_image_name_in_acr(resource_name["acrName"], DOCKER_IMAGE_NAME),
            component_name,
            script,
            env,
            volumes,
        )
        manifest = k8s_manifest_generator.get_job_manifest(
            resource_name["userPoolName"],
            component_name,
            container_spec,
            volumes,
        )
        k8s_ops.create_job(manifest, job_name)


def remove_jobs(job_names: str, **kwargs):
    if not os.path.isfile(DEPLOYMENT_CONF_PATH):
        logger.error_red(NO_DEPLOYMENT_MSG)
        return

    k8s_ops.load_config()
    for job_name in job_names:
        local_job_path = get_local_job_path(job_name)
        if not os.path.isdir(local_job_path):
            logger.error_red(NO_JOB_MSG.format(job_name))
            return

        k8s_ops.delete_job(job_name)


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


def exit(**kwargs):
    try:
        with open(DEPLOYMENT_CONF_PATH, "r") as fp:
            deployment_conf = json.load(fp)
    except FileNotFoundError:
        logger.error(NO_DEPLOYMENT_MSG)
        return

    name = deployment_conf["name"]
    set_env_credentials(LOCAL_ROOT, f"sp-{name}")
    delete_resource_group(deployment_conf["subscription"], deployment_conf["resource_group"])
