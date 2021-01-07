# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import yaml

from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
from maro.cli.grass.executors.grass_on_premises_executor import GrassOnPremisesExecutor
from maro.utils.exception.cli_exception import BadRequestError, FileOperationError, InvalidDeploymentTemplateError


def create(deployment_path: str, **kwargs):
    try:
        with open(deployment_path, "r") as fr:
            create_deployment = yaml.safe_load(fr)
        if create_deployment["mode"] == "grass/azure":
            GrassAzureExecutor.create(create_deployment=create_deployment)
        elif create_deployment["mode"] == "grass/on-premises":
            GrassOnPremisesExecutor.build_cluster_details(create_deployment=create_deployment)
            executor = GrassOnPremisesExecutor(cluster_name=create_deployment["name"])
            executor.create()
        else:
            raise BadRequestError(f"Unsupported operation in mode '{create_deployment['mode']}'.")
    except KeyError as e:
        raise InvalidDeploymentTemplateError(f"Missing key '{e.args[0]}'.")
    except FileNotFoundError:
        raise FileOperationError("Invalid template file path.")
