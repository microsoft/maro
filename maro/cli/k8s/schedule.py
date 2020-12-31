# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.k8s.executors.k8s_aks_executor import K8sAksExecutor
from maro.cli.utils.checkers import check_details_validity
from maro.cli.utils.details_reader import DetailsReader
from maro.cli.utils.lock import lock
from maro.utils.exception.cli_exception import BadRequestError


@check_details_validity
@lock
def start_schedule(cluster_name: str, deployment_path: str, **kwargs):
    # Load details
    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "k8s/aks":
        executor = K8sAksExecutor(cluster_name=cluster_name)
        executor.start_schedule(
            deployment_path=deployment_path
        )
    else:
        raise BadRequestError(f"Unsupported command in mode '{cluster_details['mode']}'.")


@check_details_validity
@lock
def stop_schedule(cluster_name: str, schedule_name: str, **kwargs):
    # Load details
    cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] == "k8s/aks":
        executor = K8sAksExecutor(cluster_name=cluster_name)
        executor.stop_schedule(
            schedule_name=schedule_name
        )
    else:
        raise BadRequestError(f"Unsupported command in mode '{cluster_details['mode']}'.")
