# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.utils.details_validity_wrapper import check_details_validity
from maro.cli.utils.operation_lock_wrapper import operation_lock


@check_details_validity
@operation_lock
def scale_node(cluster_name: str, replicas: int, node_size: str, **kwargs):
    # Late import.
    from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
    from maro.cli.utils.details_reader import DetailsReader
    from maro.utils.exception.cli_exception import BadRequestError

    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.scale_node(replicas=replicas, node_size=node_size)
    else:
        raise BadRequestError(f"Unsupported operation in mode '{cluster_details['mode']}'.")


@check_details_validity
@operation_lock
def start_node(cluster_name: str, replicas: int, node_size: str, **kwargs):
    # Late import.
    from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
    from maro.cli.utils.details_reader import DetailsReader
    from maro.utils.exception.cli_exception import BadRequestError

    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.start_node(replicas=replicas, node_size=node_size)
    else:
        raise BadRequestError(f"Unsupported operation in mode '{cluster_details['mode']}'.")


@check_details_validity
@operation_lock
def stop_node(cluster_name: str, replicas: int, node_size: str, **kwargs):
    # Late import.
    from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
    from maro.cli.utils.details_reader import DetailsReader
    from maro.utils.exception.cli_exception import BadRequestError

    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.stop_node(replicas=replicas, node_size=node_size)
    else:
        raise BadRequestError(f"Unsupported operation in mode '{cluster_details['mode']}'.")


@check_details_validity
@operation_lock
def list_node(cluster_name: str, **kwargs):
    # Late import.
    from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
    from maro.cli.grass.executors.grass_on_premises_executor import GrassOnPremisesExecutor
    from maro.cli.utils.details_reader import DetailsReader
    from maro.utils.exception.cli_exception import BadRequestError

    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.list_node()
    elif cluster_details["mode"] == "grass/on-premises":
        executor = GrassOnPremisesExecutor(cluster_name=cluster_name)
        executor.list_node()
    else:
        raise BadRequestError(f"Unsupported operation in mode '{cluster_details['mode']}'.")


def join_cluster(deployment_path: str, **kwargs):
    # Late import.
    import yaml

    from maro.cli.grass.executors.grass_on_premises_executor import GrassOnPremisesExecutor
    from maro.utils.exception.cli_exception import BadRequestError, FileOperationError, InvalidDeploymentTemplateError

    try:
        with open(deployment_path, "r") as fr:
            join_cluster_deployment = yaml.safe_load(stream=fr)
        if join_cluster_deployment["mode"] == "grass/on-premises":
            GrassOnPremisesExecutor.join_cluster(join_cluster_deployment=join_cluster_deployment)
        else:
            raise BadRequestError(f"Unsupported operation in mode '{join_cluster_deployment['mode']}'.")
    except KeyError as e:
        raise InvalidDeploymentTemplateError(f"Missing key '{e.args[0]}'.")
    except FileNotFoundError:
        raise FileOperationError("Invalid template file path.")


def leave_cluster(deployment_path: str, **kwargs):
    # Late import.
    import yaml

    from maro.cli.grass.executors.grass_on_premises_executor import GrassOnPremisesExecutor
    from maro.utils.exception.cli_exception import BadRequestError, FileOperationError, InvalidDeploymentTemplateError

    try:
        if not deployment_path:
            GrassOnPremisesExecutor.leave(leave_cluster_deployment={})
        else:
            with open(deployment_path, "r") as fr:
                leave_cluster_deployment = yaml.safe_load(stream=fr)
            if leave_cluster_deployment["mode"] == "grass/on-premises":
                GrassOnPremisesExecutor.leave(leave_cluster_deployment=leave_cluster_deployment)
            else:
                raise BadRequestError(f"Unsupported operation in mode '{leave_cluster_deployment['mode']}'.")
    except KeyError as e:
        raise InvalidDeploymentTemplateError(f"Missing key '{e.args[0]}'.")
    except FileNotFoundError:
        raise FileOperationError("Invalid template file path.")
