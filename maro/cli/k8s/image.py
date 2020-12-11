# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.k8s.executors.k8s_aks_executor import K8sAksExecutor
from maro.cli.utils.checkers import check_details_validity
from maro.cli.utils.details import load_cluster_details
from maro.cli.utils.lock import lock
from maro.utils.exception.cli_exception import BadRequestError


@check_details_validity
@lock
def push_image(cluster_name: str, image_name: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "k8s/aks":
        executor = K8sAksExecutor(cluster_name=cluster_name)
        executor.push_image(
            image_name=image_name
        )
    else:
        raise BadRequestError(f"Unsupported command in mode '{cluster_details['mode']}'.")


@check_details_validity
@lock
def list_image(cluster_name: str, **kwargs):
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "k8s/aks":
        executor = K8sAksExecutor(cluster_name=cluster_name)
        executor.list_image()
    else:
        raise BadRequestError(f"Unsupported command in mode '{cluster_details['mode']}'.")
