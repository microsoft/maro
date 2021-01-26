# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import datetime
import json

from maro.cli.utils.subprocess import Subprocess
from maro.utils.exception.cli_exception import CommandExecutionError, DeploymentError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class AzureController:
    """A wrapper class for Azure CLI.

    Exec Azure CLI command to see more details.
    """

    # Account related

    @staticmethod
    def set_subscription(subscription: str) -> None:
        command = f"az account set --subscription {subscription}"
        _ = Subprocess.run(command=command)

    # Resource Group related

    @staticmethod
    def get_resource_group(resource_group: str) -> dict:
        command = f"az group show --name {resource_group}"
        try:
            return_str = Subprocess.run(command=command)
            return json.loads(return_str)
        except CommandExecutionError:
            return {}

    @staticmethod
    def create_resource_group(resource_group: str, location: str) -> None:
        command = f"az group create --name {resource_group} --location {location}"
        _ = Subprocess.run(command=command)

    @staticmethod
    def delete_resource_group(resource_group: str) -> None:
        command = f"az group delete --yes --name {resource_group}"
        _ = Subprocess.run(command=command)

    # Resource related

    @staticmethod
    def list_resources(resource_group: str) -> list:
        command = f"az resource list -g {resource_group}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    @staticmethod
    def delete_resources(resource_ids: list) -> None:
        command = f"az resource delete --ids {' '.join(resource_ids)}"
        _ = Subprocess.run(command=command)

    # Deployment related

    @staticmethod
    def start_deployment(
        resource_group: str,
        deployment_name: str,
        template_file_path: str, parameters_file_path: str
    ) -> None:
        command = (
            f"az deployment group create -g {resource_group} --name {deployment_name} "
            f"--template-file {template_file_path} --parameters {parameters_file_path}"
        )
        try:
            _ = Subprocess.run(command=command)
        except CommandExecutionError as e:
            error = json.loads(AzureController._get_valid_json(e.get_message()))["error"]
            raise DeploymentError(error["message"])

    @staticmethod
    def delete_deployment(resource_group: str, deployment_name: str) -> None:
        command = f"az deployment group delete -g {resource_group} --name {deployment_name}"
        _ = Subprocess.run(command=command)

    # VM related

    @staticmethod
    def list_ip_addresses(resource_group: str, vm_name: str) -> list:
        command = f"az vm list-ip-addresses -g {resource_group} --name {vm_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    @staticmethod
    def start_vm(resource_group: str, vm_name: str) -> None:
        command = f"az vm start -g {resource_group} --name {vm_name}"
        _ = Subprocess.run(command=command)

    @staticmethod
    def stop_vm(resource_group: str, vm_name: str) -> None:
        command = f"az vm stop -g {resource_group} --name {vm_name}"
        _ = Subprocess.run(command=command)

    @staticmethod
    def list_vm_sizes(location: str) -> list:
        command = f"az vm list-sizes -l {location}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    @staticmethod
    def deallocate_vm(resource_group: str, vm_name: str) -> None:
        command = f"az vm deallocate --resource-group {resource_group} --name {vm_name}"
        _ = Subprocess.run(command=command)

    @staticmethod
    def generalize_vm(resource_group: str, vm_name: str) -> None:
        command = f"az vm generalize --resource-group {resource_group} --name {vm_name}"
        _ = Subprocess.run(command=command)

    # Image related

    @staticmethod
    def create_image_from_vm(resource_group: str, image_name: str, vm_name: str) -> None:
        command = f"az image create --resource-group {resource_group} --name {image_name} --source {vm_name}"
        _ = Subprocess.run(command=command)

    @staticmethod
    def get_image_resource_id(resource_group: str, image_name: str) -> str:
        command = f"az image show --resource-group {resource_group} --name {image_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)["id"]

    # AKS related

    @staticmethod
    def load_aks_context(resource_group: str, aks_name: str) -> None:
        command = f"az aks get-credentials -g {resource_group} --name {aks_name}"
        _ = Subprocess.run(command=command)

    @staticmethod
    def get_aks(resource_group: str, aks_name: str) -> dict:
        command = f"az aks show -g {resource_group} -n {aks_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    @staticmethod
    def attach_acr(resource_group: str, aks_name: str, acr_name: str) -> None:
        command = f"az aks update -g {resource_group} --name {aks_name} --attach-acr {acr_name}"
        _ = Subprocess.run(command=command)

    @staticmethod
    def list_nodepool(resource_group: str, aks_name: str) -> list:
        command = f"az aks nodepool list -g {resource_group} --cluster-name {aks_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    @staticmethod
    def add_nodepool(resource_group: str, aks_name: str, nodepool_name: str, node_count: int, node_size: str) -> None:
        command = (
            f"az aks nodepool add "
            f"-g {resource_group} "
            f"--cluster-name {aks_name} "
            f"--name {nodepool_name} "
            f"--node-count {node_count} "
            f"--node-vm-size {node_size}"
        )
        _ = Subprocess.run(command=command)

    @staticmethod
    def scale_nodepool(resource_group: str, aks_name: str, nodepool_name: str, node_count: int) -> None:
        command = (
            f"az aks nodepool scale "
            f"-g {resource_group} "
            f"--cluster-name {aks_name} "
            f"--name {nodepool_name} "
            f"--node-count {node_count}"
        )
        _ = Subprocess.run(command=command)

    # ACR related

    @staticmethod
    def login_acr(acr_name: str) -> None:
        command = f"az acr login --name {acr_name}"
        _ = Subprocess.run(command=command)

    @staticmethod
    def list_acr_repositories(acr_name: str) -> list:
        command = f"az acr repository list -n {acr_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    # Storage account related

    @staticmethod
    def get_storage_account_keys(resource_group: str, storage_account_name: str) -> dict:
        command = f"az storage account keys list -g {resource_group} --account-name {storage_account_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    @staticmethod
    def get_storage_account_sas(
        account_name: str,
        services: str = "bqtf",
        resource_types: str = "sco",
        permissions: str = "rwdlacup",
        expiry: str = (datetime.datetime.utcnow() + datetime.timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    ) -> str:
        command = (
            f"az storage account generate-sas --account-name {account_name} --services {services} "
            f"--resource-types {resource_types} --permissions {permissions} --expiry {expiry}"
        )
        sas_str = Subprocess.run(command=command).strip("\n").replace('"', "")
        logger.debug(sas_str)
        return sas_str

    @staticmethod
    def get_connection_string(storage_account_name: str) -> str:
        """Get the connection string for a storage account.

        Args:
            storage_account_name: The storage account name.

        Returns:
            str: Connection string.
        """
        command = f"az storage account show-connection-string --name {storage_account_name}"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)["connectionString"]

    # Utils

    @staticmethod
    def get_version() -> dict:
        command = "az version"
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)

    @staticmethod
    def _get_valid_json(message: str) -> str:
        left_idx = message.find("{")
        right_idx = message.rindex("}")
        return message[left_idx:right_idx + 1]
