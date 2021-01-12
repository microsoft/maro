# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import yaml

from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
from maro.cli.grass.executors.grass_on_premises_executor import GrassOnPremisesExecutor
from maro.cli.utils.details_reader import DetailsReader
from maro.cli.utils.details_validity_wrapper import check_details_validity
from maro.cli.utils.operation_lock_wrapper import operation_lock
from maro.utils.exception.cli_exception import BadRequestError, FileOperationError, InvalidDeploymentTemplateError


@check_details_validity
@operation_lock
def scale_node(cluster_name: str, replicas: int, node_size: str, **kwargs):
    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.scale_node(replicas=replicas, node_size=node_size)
    else:
        raise BadRequestError(f"Unsupported operation in mode '{cluster_details['mode']}'.")


@check_details_validity
@operation_lock
def start_node(cluster_name: str, replicas: int, node_size: str, **kwargs):
    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.start_node(replicas=replicas, node_size=node_size)
    else:
        raise BadRequestError(f"Unsupported operation in mode '{cluster_details['mode']}'.")


@check_details_validity
@operation_lock
def stop_node(cluster_name: str, replicas: int, node_size: str, **kwargs):
    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.stop_node(replicas=replicas, node_size=node_size)
    else:
        raise BadRequestError(f"Unsupported operation in mode '{cluster_details['mode']}'.")


@check_details_validity
@operation_lock
def list_node(cluster_name: str, **kwargs):
    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "grass/azure":
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.list_node()
    elif cluster_details["mode"] == "grass/on-premises":
        executor = GrassOnPremisesExecutor(cluster_name=cluster_name)
        executor.list_node()
    else:
        raise BadRequestError(f"Unsupported operation in mode '{cluster_details['mode']}'.")


def node_join(deployment_path: str, **kwargs):
    try:
        with open(deployment_path, "r") as fr:
            join_node_deployment = yaml.safe_load(fr)
        if join_node_deployment["mode"] == "grass/on-premises":
            GrassOnPremisesExecutor.join_node(join_node_deployment=join_node_deployment)
        else:
            raise BadRequestError(f"Unsupported operation in mode '{join_node_deployment['mode']}'.")
    except KeyError as e:
        raise InvalidDeploymentTemplateError(f"Missing key '{e.args[0]}'.")
    except FileNotFoundError:
        raise FileOperationError("Invalid template file path.")


def node_leave(deployment_path: str, **kwargs):
    try:
        if not deployment_path:
            GrassOnPremisesExecutor.leave(leave_node_deployment={})
        else:
            with open(deployment_path, "r") as fr:
                leave_node_deployment = yaml.safe_load(fr)
            if leave_node_deployment["mode"] == "grass/on-premises":
                GrassOnPremisesExecutor.leave(leave_node_deployment=leave_node_deployment)
            else:
                raise BadRequestError(f"Unsupported operation in mode '{leave_node_deployment['mode']}'.")
    except KeyError as e:
        raise InvalidDeploymentTemplateError(f"Missing key '{e.args[0]}'.")
    except FileNotFoundError:
        raise FileOperationError("Invalid template file path.")
