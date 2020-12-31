# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
from maro.cli.grass.executors.grass_on_premises_executor import GrassOnPremisesExecutor
from maro.cli.utils.details_reader import DetailsReader
from maro.cli.utils.details_validity_wrapper import check_details_validity
from maro.cli.utils.lock import lock
from maro.utils.exception.cli_exception import BadRequestError


@check_details_validity
@lock
def start_schedule(cluster_name: str, deployment_path: str, **kwargs):
    # Load details
    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.start_schedule(deployment_path=deployment_path)
    elif cluster_details["mode"] == "grass/on-premises":
        executor = GrassOnPremisesExecutor(cluster_name=cluster_name)
        executor.start_schedule(deployment_path=deployment_path)
    else:
        raise BadRequestError(f"Unsupported command in mode '{cluster_details['mode']}'.")


@check_details_validity
@lock
def stop_schedule(cluster_name: str, schedule_name: str, **kwargs):
    # Load details
    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.stop_schedule(schedule_name=schedule_name)
    elif cluster_details["mode"] == "grass/on-premises":
        executor = GrassOnPremisesExecutor(cluster_name=cluster_name)
        executor.stop_schedule(schedule_name=schedule_name)
    else:
        raise BadRequestError(f"Unsupported command in mode '{cluster_details['mode']}'.")
