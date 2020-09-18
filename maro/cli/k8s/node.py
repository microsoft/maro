# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.k8s.executors.k8s_azure_executor import K8sAzureExecutor
from maro.cli.utils.checkers import check_details_validity
from maro.cli.utils.details import load_cluster_details
from maro.cli.utils.lock import lock


@check_details_validity(mode='k8s')
@lock
def scale_node(cluster_name: str, replicas: int, node_size: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = K8sAzureExecutor(cluster_name=cluster_name)
        executor.scale_node(
            replicas=replicas,
            node_size=node_size
        )


@check_details_validity(mode='k8s')
@lock
def list_node(cluster_name: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = K8sAzureExecutor(cluster_name=cluster_name)
        executor.list_node()
