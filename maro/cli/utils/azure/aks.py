# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess

from azure.identity import DefaultAzureCredential
from azure.mgmt.authorization import AuthorizationManagementClient
from azure.mgmt.containerservice import ContainerServiceClient

from maro.cli.utils.subprocess import Subprocess


def get_container_service_client(subscription: str):
    return ContainerServiceClient(DefaultAzureCredential(), subscription)


def get_authorization_client(subscription: str):
    return AuthorizationManagementClient()


def load_aks_context(resource_group: str, aks_name: str) -> None:
    command = f"az aks get-credentials -g {resource_group} --name {aks_name}"
    _ = Subprocess.run(command=command)


def get_aks(subscription: str, resource_group: str, aks_name: str) -> dict:
    container_service_client = get_container_service_client(subscription)
    return container_service_client.managed_clusters.get(resource_group, aks_name)


def attach_acr(resource_group: str, aks_name: str, acr_name: str) -> None:
    subprocess.run(f"az aks update -g {resource_group} -n {aks_name} --attach-acr {acr_name}".split())


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


def scale_nodepool(resource_group: str, aks_name: str, nodepool_name: str, node_count: int) -> None:
    command = (
        f"az aks nodepool scale "
        f"-g {resource_group} "
        f"--cluster-name {aks_name} "
        f"--name {nodepool_name} "
        f"--node-count {node_count}"
    )
    _ = Subprocess.run(command=command)
