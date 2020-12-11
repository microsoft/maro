# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import yaml

from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
from maro.utils.exception.cli_exception import InvalidDeploymentTemplateError, FileOperationError


def create(deployment_path: str, **kwargs):
    try:
        with open(deployment_path, 'r') as fr:
            create_deployment = yaml.safe_load(fr)

        if create_deployment['mode'] == 'grass':
            if create_deployment['cloud']['infra'] == 'azure':
                GrassAzureExecutor.build_cluster_details(create_deployment=create_deployment)
                executor = GrassAzureExecutor(cluster_name=create_deployment['name'])
                executor.create()
            else:
                raise InvalidDeploymentTemplateError(
                    f"Deployment is broken: Invalid infra: {create_deployment['cloud']['infra']}")
        else:
            raise InvalidDeploymentTemplateError(f"Deployment is broken: Invalid mode: {create_deployment['mode']}")
    except KeyError as e:
        raise InvalidDeploymentTemplateError(f"Deployment is broken: Missing key: '{e.args[0]}'")
    except FileNotFoundError:
        raise FileOperationError(f"Invalid template file path.")
