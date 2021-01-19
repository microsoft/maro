# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.utils.details_validity_wrapper import check_details_validity
from maro.cli.utils.operation_lock_wrapper import operation_lock


@check_details_validity
@operation_lock
def push_image(
    cluster_name: str, image_name: str, image_path: str, remote_context_path: str, remote_image_name: str,
    **kwargs
):
    # Late imports.
    from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
    from maro.cli.grass.executors.grass_on_premises_executor import GrassOnPremisesExecutor
    from maro.cli.utils.details_reader import DetailsReader
    from maro.utils.exception.cli_exception import BadRequestError

    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.push_image(
            image_name=image_name,
            image_path=image_path,
            remote_context_path=remote_context_path,
            remote_image_name=remote_image_name
        )
    elif cluster_details["mode"] == "grass/on-premises":
        executor = GrassOnPremisesExecutor(cluster_name=cluster_name)
        executor.push_image(
            image_name=image_name,
            image_path=image_path,
            remote_context_path=remote_context_path,
            remote_image_name=remote_image_name
        )
    else:
        raise BadRequestError(f"Unsupported operation in mode '{cluster_details['mode']}'.")
