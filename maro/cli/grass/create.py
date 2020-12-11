# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import yaml

from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
from maro.utils.exception.cli_exception import InvalidDeploymentTemplateError, FileOperationError, BadRequestError


def create(deployment_path: str, **kwargs):
    try:
        with open(deployment_path, "r") as fr:
            create_deployment = yaml.safe_load(fr)
        if create_deployment["mode"] == "grass/azure":
            GrassAzureExecutor.build_cluster_details(create_deployment=create_deployment)
            executor = GrassAzureExecutor(cluster_name=create_deployment["name"])
            executor.create()
        else:
            raise BadRequestError(f"Unsupported command in mode '{create_deployment['mode']}'.")
    except KeyError as e:
        raise InvalidDeploymentTemplateError(f"Missing key '{e.args[0]}'.")
    except FileNotFoundError:
        raise FileOperationError(f"Invalid template file path.")
