# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import datetime
import json

from maro.cli.utils.subprocess import SubProcess
from maro.utils.exception.cli_exception import CommandError, DeploymentError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class AzureExecutor:

    # Account related

    @staticmethod
    def set_subscription(subscription: str):
        command = f"az account set --subscription {subscription}"
        _ = SubProcess.run(command)

    # Resource Group related

    @staticmethod
    def get_resource_group(resource_group: str):
        command = f"az group show --name {resource_group}"
        try:
            return_str = SubProcess.run(command)
            return json.loads(return_str)
        except CommandError:
            return None

    @staticmethod
    def create_resource_group(resource_group: str, location: str):
        command = f"az group create --name {resource_group} --location {location}"
        _ = SubProcess.run(command)

    # Resource related

    @staticmethod
    def list_resources(resource_group: str):
        command = f"az resource list -g {resource_group}"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    @staticmethod
    def delete_resources(resources: list):
        command = f"az resource delete --ids {' '.join(resources)}"
        _ = SubProcess.run(command)

    # Deployment related

    @staticmethod
    def start_deployment(resource_group: str, deployment_name: str, template_file: str, parameters_file: str):
        command = f"az deployment group create -g {resource_group} --name {deployment_name} " \
                  f"--template-file {template_file} --parameters {parameters_file}"
        try:
            _ = SubProcess.run(command)
        except CommandError as e:
            error = json.loads(AzureExecutor._get_valid_json(e.get_message()))['error']
            raise DeploymentError(error['message'])

    @staticmethod
    def delete_deployment(resource_group: str, deployment_name: str):
        command = f"az deployment group delete -g {resource_group} --name {deployment_name}"
        _ = SubProcess.run(command)

    # VM related

    @staticmethod
    def list_ip_addresses(resource_group: str, vm_name: str):
        command = f"az vm list-ip-addresses -g {resource_group} --name {vm_name}"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    @staticmethod
    def start_vm(resource_group: str, vm_name: str):
        command = f"az vm start -g {resource_group} --name {vm_name}"
        _ = SubProcess.run(command)

    @staticmethod
    def stop_vm(resource_group: str, vm_name: str):
        command = f"az vm stop -g {resource_group} --name {vm_name}"
        _ = SubProcess.run(command)

    @staticmethod
    def list_vm_sizes(location: str):
        command = f"az vm list-sizes -l {location}"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    @staticmethod
    def list_skus(vm_size: str, location: str):
        command = f"az vm list-skus -l {location} --all --size {vm_size}"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    @staticmethod
    def get_sku(vm_size: str, location: str):
        skus = AzureExecutor.list_skus(vm_size=vm_size, location=location)
        for sku in skus:
            if sku["name"] == vm_size:
                return sku
        logger.warning_yellow(f"SKU of {vm_size} is not found")
        return None

    # AKS related

    @staticmethod
    def load_aks_context(resource_group: str, aks_name: str):
        command = f'az aks get-credentials -g {resource_group} --name {aks_name}'
        _ = SubProcess.run(command)

    @staticmethod
    def get_aks(resource_group: str, aks_name: str):
        command = f"az aks show -g {resource_group} -n {aks_name}"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    @staticmethod
    def attach_acr(resource_group: str, aks_name: str, acr_name: str):
        command = f'az aks update -g {resource_group} --name {aks_name} --attach-acr {acr_name}'
        _ = SubProcess.run(command)

    @staticmethod
    def list_nodepool(resource_group: str, aks_name: str):
        command = f'az aks nodepool list -g {resource_group} --cluster-name {aks_name}'
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    @staticmethod
    def add_nodepool(resource_group: str, aks_name: str, nodepool_name: str, node_count: int, node_size: str):
        command = f"az aks nodepool add " \
                  f"-g {resource_group} " \
                  f"--cluster-name {aks_name} " \
                  f"--name {nodepool_name} " \
                  f"--node-count {node_count} " \
                  f"--node-vm-size {node_size}"
        _ = SubProcess.run(command)

    @staticmethod
    def scale_nodepool(resource_group: str, aks_name: str, nodepool_name: str, node_count: int):
        command = f"az aks nodepool scale " \
                  f"-g {resource_group} " \
                  f"--cluster-name {aks_name} " \
                  f"--name {nodepool_name} " \
                  f"--node-count {node_count}"
        _ = SubProcess.run(command)

    # ACR related

    @staticmethod
    def login_acr(acr_name: str):
        command = f"az acr login --name {acr_name}"
        _ = SubProcess.run(command)

    @staticmethod
    def list_acr_repositories(acr_name: str):
        command = f"az acr repository list -n {acr_name}"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    # Storage account related

    @staticmethod
    def get_storage_account_keys(resource_group: str, storage_account_name: str):
        command = f'az storage account keys list -g {resource_group} --account-name {storage_account_name}'
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    @staticmethod
    def get_storage_account_sas(account_name: str,
                                services: str = 'bqtf',
                                resource_types: str = 'sco',
                                permissions: str = 'rwdlacup',
                                expiry: str = (datetime.datetime.utcnow() +
                                               datetime.timedelta(days=365)).strftime('%Y-%m-%dT%H:%M:%S') + 'Z'):
        command = f'az storage account generate-sas --account-name {account_name} --services {services} ' \
                  f'--resource-types {resource_types} --permissions {permissions} --expiry {expiry}'
        sas_str = SubProcess.run(command=command).strip('\n').replace('"', '')
        logger.debug(sas_str)
        return sas_str

    # Utils

    @staticmethod
    def get_version():
        command = "az version"
        return_str = SubProcess.run(command)
        return json.loads(return_str)

    @staticmethod
    def _get_valid_json(message: str):
        left_idx = message.find('{')
        right_idx = message.rindex('}')
        return message[left_idx:right_idx + 1]
