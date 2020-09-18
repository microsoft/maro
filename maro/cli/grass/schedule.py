# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
from maro.cli.utils.checkers import check_details_validity
from maro.cli.utils.details import load_cluster_details
from maro.cli.utils.lock import lock


@check_details_validity(mode='grass')
@lock
def start_schedule(cluster_name: str, deployment_path: str, **kwargs):
    # Load details
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.start_schedule(deployment_path=deployment_path)


@check_details_validity(mode='grass')
@lock
def stop_schedule(cluster_name: str, schedule_name: str, **kwargs):
    # Load details
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.stop_schedule(schedule_name=schedule_name)
