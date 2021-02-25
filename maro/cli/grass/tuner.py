# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.utils.details_validity_wrapper import check_details_validity
from maro.cli.utils.operation_lock_wrapper import operation_lock


@check_details_validity
@operation_lock
def start_tuner(cluster_name: str, deployment_path: str, **kwargs):
    # Late import.
    from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
    from maro.cli.grass.executors.grass_local_executor import GrassLocalExecutor
    # from maro.cli.grass.executors.grass_on_premises_executor import GrassOnPremisesExecutor
    from maro.cli.utils.details_reader import DetailsReader
    from maro.utils.exception.cli_exception import BadRequestError

    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.start_tuner(deployment_path=deployment_path)
    elif cluster_details["mode"] == "grass/local":
        executor = GrassLocalExecutor(cluster_name=cluster_name)
        executor.start_tuner(deployment_path=deployment_path)
    elif cluster_details["mode"] == "grass/on-premises":
        # executor = GrassOnPremisesExecutor(cluster_name=cluster_name)
        # executor.start_tuner(deployment_path=deployment_path)
        raise BadRequestError(f"Not implemented operation in mode '{cluster_details['mode']}'.")
    else:
        raise BadRequestError(f"Unsupported operation in mode '{cluster_details['mode']}'.")


@check_details_validity
@operation_lock
def stop_tuner(cluster_name: str, tuner_name: str, **kwargs):
    # Late import.
    from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
    from maro.cli.grass.executors.grass_local_executor import GrassLocalExecutor
    # from maro.cli.grass.executors.grass_on_premises_executor import GrassOnPremisesExecutor
    from maro.cli.utils.details_reader import DetailsReader
    from maro.utils.exception.cli_exception import BadRequestError

    # Load details
    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.stop_tuner(tuner_name=tuner_name)
    elif cluster_details["mode"] == "grass/local":
        executor = GrassLocalExecutor(cluster_name=cluster_name)
        executor.stop_tuner(tuner_name=tuner_name)
    elif cluster_details["mode"] == "grass/on-premises":
        # executor = GrassOnPremisesExecutor(cluster_name=cluster_name)
        # executor.stop_tuner(stop_tuner=stop_tuner)
        raise BadRequestError(f"Not implemented operation in mode '{cluster_details['mode']}'.")
    else:
        raise BadRequestError(f"Unsupported operation in mode '{cluster_details['mode']}'.")
