# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import yaml

from maro.cli.k8s.executors.k8s_aks_executor import K8sAksExecutor
from maro.utils.exception.cli_exception import BadRequestError, FileOperationError, InvalidDeploymentTemplateError


def create(deployment_path: str, **kwargs):
    try:
        with open(deployment_path, 'r') as fr:
            create_deployment = yaml.safe_load(fr)
        if create_deployment["mode"] == "k8s/aks":
            K8sAksExecutor.build_cluster_details(create_deployment=create_deployment)
            executor = K8sAksExecutor(cluster_name=create_deployment["name"])
            executor.create()
        else:
            raise BadRequestError(f"Unsupported command in mode '{create_deployment['mode']}'.")
    except KeyError as e:
        raise InvalidDeploymentTemplateError(f"Missing key '{e.args[0]}'.")
    except FileNotFoundError:
        raise FileOperationError("Invalid template file path.")
