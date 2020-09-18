# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.k8s.executors.k8s_azure_executor import K8sAzureExecutor
from maro.cli.utils.checkers import check_details_validity
from maro.cli.utils.details import load_cluster_details
from maro.cli.utils.lock import lock


@check_details_validity(mode='k8s')
@lock
def push_data(cluster_name: str, local_path: str, remote_dir: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = K8sAzureExecutor(cluster_name=cluster_name)
        executor.push_data(
            local_path=local_path,
            remote_dir=remote_dir
        )


@check_details_validity(mode='k8s')
@lock
def pull_data(cluster_name: str, local_dir: str, remote_path: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = K8sAzureExecutor(cluster_name=cluster_name)
        executor.pull_data(
            local_dir=local_dir,
            remote_path=remote_path
        )


@check_details_validity(mode='k8s')
@lock
def remove_data(cluster_name: str, remote_path: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = K8sAzureExecutor(cluster_name=cluster_name)
        executor.remove_data(
            remote_path=remote_path
        )
