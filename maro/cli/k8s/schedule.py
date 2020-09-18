# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.k8s.executors.k8s_azure_executor import K8sAzureExecutor
from maro.cli.utils.checkers import check_details_validity
from maro.cli.utils.details import load_cluster_details
from maro.cli.utils.lock import lock


@check_details_validity(mode='k8s')
@lock
def start_schedule(cluster_name: str, deployment_path: str, **kwargs):
    # Load details
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = K8sAzureExecutor(cluster_name=cluster_name)
        executor.start_schedule(
            deployment_path=deployment_path
        )


@check_details_validity(mode='k8s')
@lock
def stop_schedule(cluster_name: str, schedule_name: str, **kwargs):
    # Load details
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = K8sAzureExecutor(cluster_name=cluster_name)
        executor.stop_schedule(
            schedule_name=schedule_name
        )
