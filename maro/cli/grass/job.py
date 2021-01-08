# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
from maro.cli.grass.executors.grass_local_executor import GrassLocalExecutor
from maro.cli.utils.checkers import check_details_validity
from maro.cli.utils.details import load_cluster_details
from maro.cli.utils.lock import lock
from maro.utils.exception.cli_exception import BadRequestError


@check_details_validity
@lock
def start_job(cluster_name: str, deployment_path: str, **kwargs):
    # Load details
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] in ["grass/azure", "grass/on-premises"]:
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.start_job(deployment_path=deployment_path)
    elif cluster_details["mode"] == "grass/local":
        executor = GrassLocalExecutor(cluster_name=cluster_name)
        executor.start_job(deployment_path=deployment_path)
    else:
        raise BadRequestError(f"Unsupported command in mode '{cluster_details['mode']}'.")


@check_details_validity
@lock
def stop_job(cluster_name: str, job_name: str, **kwargs):
    # Load details
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] in ["grass/azure", "grass/on-premises"]:
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.stop_job(job_name=job_name)
    elif cluster_details["mode"] == "grass/local":
        executor = GrassLocalExecutor(cluster_name=cluster_name)
        executor.stop_job(job_name=job_name)
    else:
        raise BadRequestError(f"Unsupported command in mode '{cluster_details['mode']}'.")


@check_details_validity
@lock
def list_job(cluster_name: str, **kwargs):
    # Load details
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] in ["grass/azure", "grass/on-premises"]:
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.list_job()
    elif cluster_details["mode"] == "grass/local":
        executor = GrassLocalExecutor(cluster_name=cluster_name)
        executor.list_job()
    else:
        raise BadRequestError(f"Unsupported command in mode '{cluster_details['mode']}'.")


@check_details_validity
@lock
def get_job_logs(cluster_name: str, job_name: str, **kwargs):
    # Load details
    cluster_details = load_cluster_details(cluster_name=cluster_name)

    if cluster_details["mode"] in ["grass/azure", "grass/on-premises"]:
        executor = GrassAzureExecutor(cluster_name=cluster_name)
        executor.get_job_logs(job_name=job_name)
    elif cluster_details["mode"] == "grass/local":
        executor = GrassLocalExecutor(cluster_name=cluster_name)
        executor.get_job_logs(job_name=job_name)
    else:
        raise BadRequestError(f"Unsupported command in mode '{cluster_details['mode']}'.")
