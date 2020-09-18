# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
from maro.cli.utils.checkers import check_details_validity
from maro.cli.utils.details import load_cluster_details
from maro.cli.utils.lock import lock


@check_details_validity(mode='grass')
@lock
def push_image(cluster_name: str, image_name: str, image_path: str, remote_context_path: str, remote_image_name: str,
               **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details['cloud']['infra'] == 'azure':
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.push_image(
            image_name=image_name,
            image_path=image_path,
            remote_context_path=remote_context_path,
            remote_image_name=remote_image_name
        )
