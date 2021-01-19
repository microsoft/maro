# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def create(deployment_path: str, **kwargs):
    # Late import.
    import yaml

    from maro.cli.k8s.executors.k8s_aks_executor import K8sAksExecutor
    from maro.utils.exception.cli_exception import BadRequestError, FileOperationError, InvalidDeploymentTemplateError

    try:
        with open(deployment_path, "r") as fr:
            create_deployment = yaml.safe_load(fr)
        if create_deployment["mode"] == "k8s/aks":
            K8sAksExecutor.create(create_deployment=create_deployment)
        else:
            raise BadRequestError(f"Unsupported operation in mode '{create_deployment['mode']}'.")
    except KeyError as e:
        raise InvalidDeploymentTemplateError(f"Missing key '{e.args[0]}'.")
    except FileNotFoundError:
        raise FileOperationError("Invalid template file path.")
