# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.k8s.executors.k8s_azure_executor import K8sAzureExecutor
from maro.cli.utils.checkers import check_details_validity
from maro.cli.utils.details import load_cluster_details
from maro.cli.utils.lock import lock


@check_details_validity(mode='k8s')
@lock
def push_image(cluster_name: str, image_name: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = K8sAzureExecutor(cluster_name=cluster_name)
        executor.push_image(
            image_name=image_name
        )


@check_details_validity(mode='k8s')
@lock
def list_image(cluster_name: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = K8sAzureExecutor(cluster_name=cluster_name)
        executor.list_image()
