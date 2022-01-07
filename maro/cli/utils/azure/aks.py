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


# def list_nodepool(resource_group: str, aks_name: str) -> list:
#     container_service_client = get_container_service_client(subscription)
#     container_service_client.container_services.
#     command = f"az aks nodepool list -g {resource_group} --cluster-name {aks_name}"
#     return_str = Subprocess.run(command=command)
#     return json.loads(return_str)


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


def create_aks_cluster(subscription: str, resource_group: str, cluster_name: str):
    container_service_client = get_container_service_client(subscription)
    parameters = ManagedCluster(
        location=location,
        dns_prefix=dns_prefix,
        kubernetes_version=kubernetes_version,
        tags=stags,
        service_principal_profile=service_principal_profile, # this needs to be a model as well
        agent_pool_profiles=agent_pools, # this needs to be a model as well
        linux_profile=linux_profile, # this needs to be a model as well
        enable_rbac=true
    )
    container_service_client.managed_clusters.cre(resource_group, cluster_name, parameters)


def aks_agentpool_add(cmd, client, resource_group_name, cluster_name, nodepool_name,
                      kubernetes_version=None,
                      zones=None,
                      enable_node_public_ip=False,
                      node_vm_size=None,
                      node_osdisk_size=0,
                      node_count=3,
                      vnet_subnet_id=None,
                      max_pods=0,
                      os_type="Linux",
                      min_count=None,
                      max_count=None,
                      enable_cluster_autoscaler=False,
                      node_taints=None,
                      tags=None,
                      labels=None,
                      mode="User",
                      no_wait=False):
    instances = client.list(resource_group_name, cluster_name)
    for agentpool_profile in instances:
        if agentpool_profile.name == nodepool_name:
            raise CLIError("Node pool {} already exists, please try a different name, "
                           "use 'aks nodepool list' to get current list of node pool".format(nodepool_name))

    taints_array = []

    if node_taints is not None:
        for taint in node_taints.split(','):
            try:
                taint = taint.strip()
                taints_array.append(taint)
            except ValueError:
                raise CLIError('Taint does not match allowed values. Expect value such as "special=true:NoSchedule".')

    if node_vm_size is None:
        if os_type.lower() == "windows":
            node_vm_size = "Standard_D2s_v3"
        else:
            node_vm_size = "Standard_DS2_v2"

    agent_pool = AgentPool(
        name=nodepool_name,
        tags=tags,
        node_labels=labels,
        count=int(node_count),
        vm_size=node_vm_size,
        os_type=os_type,
        storage_profile=ContainerServiceStorageProfileTypes.managed_disks,
        vnet_subnet_id=vnet_subnet_id,
        agent_pool_type="VirtualMachineScaleSets",
        max_pods=int(max_pods) if max_pods else None,
        orchestrator_version=kubernetes_version,
        availability_zones=zones,
        enable_node_public_ip=enable_node_public_ip,
        node_taints=taints_array,
        mode=mode
    )

    _check_cluster_autoscaler_flag(enable_cluster_autoscaler, min_count, max_count, node_count, agent_pool)

    if node_osdisk_size:
        agent_pool.os_disk_size_gb = int(node_osdisk_size)

    return sdk_no_wait(no_wait, client.create_or_update, resource_group_name, cluster_name, nodepool_name, agent_pool)