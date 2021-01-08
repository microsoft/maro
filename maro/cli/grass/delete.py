# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
from maro.cli.grass.executors.grass_on_premises_executor import GrassOnPremisesExecutor
from maro.cli.grass.executors.grass_local_executor import GrassLocalExecutor
from maro.cli.utils.checkers import check_details_validity
from maro.cli.utils.details import load_cluster_details
from maro.cli.utils.lock import lock
from maro.utils.exception.cli_exception import BadRequestError


@check_details_validity
@lock
def delete(cluster_name: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.delete()
    elif cluster_details["mode"] == "grass/on-premises":
        executor = GrassOnPremisesExecutor(cluster_name=cluster_name)
        executor.delete()
    elif cluster_details["mode"] == "grass/local":
        executor = GrassLocalExecutor(cluster_name=cluster_name)
        executor.delete()
    else:
        raise BadRequestError(f"Unsupported command in mode '{cluster_details['mode']}'.")
