# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
from maro.cli.grass.executors.grass_on_premises_executor import GrassOnPremisesExecutor
from maro.cli.utils.checkers import check_details_validity
from maro.cli.utils.details import load_cluster_details
from maro.cli.utils.lock import lock


@check_details_validity(mode='grass')
@lock
def scale_node(cluster_name: str, replicas: int, node_size: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.scale_node(replicas=replicas, node_size=node_size)


@check_details_validity(mode='grass')
@lock
def start_node(cluster_name: str, replicas: int, node_size: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass":
        if cluster_details['cloud']['infra'] == 'azure':
            executor = GrassAzureExecutor(cluster_name=cluster_name)
            executor.start_node(replicas=replicas, node_size=node_size)


@check_details_validity(mode='grass')
@lock
def stop_node(cluster_name: str, replicas: int, node_size: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.stop_node(replicas=replicas, node_size=node_size)


@check_details_validity(mode='grass')
@lock
def list_node(cluster_name: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.list_node()


@check_details_validity(mode="grass/on-premises")
@lock
def node_join(cluster_name: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)
    executor = GrassOnPremisesExecutor(cluster_name)
    executor.node_join_cluster()
    pass

@check_details_validity(mode="grass/on-premises")
@lock
def node_leave(cluster_name: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)
    pass
