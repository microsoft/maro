# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import requests
from flask import Blueprint, jsonify, request

from ..objects import redis_controller, local_cluster_details, local_master_details

# Flask related.

blueprint = Blueprint(name="jobs", import_name=__name__)
URL_PREFIX = "/v1/jobs"


# Api functions.

@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
def list_jobs():
    """List the jobs in the cluster.

    Returns:
        None.
    """

    name_to_job_details = redis_controller.get_name_to_job_details(cluster_name=local_cluster_details["name"])
    return jsonify(list(name_to_job_details.values()))


@blueprint.route(f"{URL_PREFIX}/<job_name>", methods=["GET"])
def get_job(job_name: str):
    """Get the job with job_name.

    Returns:
        None.
    """

    job_details = redis_controller.get_job_details(
        cluster_name=local_cluster_details["name"],
        job_name=job_name
    )
    return job_details


@blueprint.route(f"{URL_PREFIX}", methods=["POST"])
def create_job():
    """Create a job.

    Returns:
        None.
    """

    job_details = request.json
    redis_controller.set_job_details(
        cluster_name=local_cluster_details["name"],
        job_name=job_details["name"],
        job_details=job_details
    )
    redis_controller.push_pending_job_ticket(
        cluster_name=local_cluster_details["name"],
        job_name=job_details["name"]
    )
    return {}


@blueprint.route(f"{URL_PREFIX}/<job_name>", methods=["DELETE"])
def delete_job(job_name: str):
    """Delete a job.

    Returns:
        None.
    """

    redis_controller.remove_pending_job_ticket(
        cluster_name=local_cluster_details["name"],
        job_name=job_name
    )
    redis_controller.push_killed_job_ticket(
        cluster_name=local_cluster_details["name"],
        job_name=job_name
    )
    redis_controller.delete_job_details(
        cluster_name=local_cluster_details["name"],
        job_name=job_name
    )
    return {}


@blueprint.route(f"{URL_PREFIX}/<job_name>:stop", methods=["POST"])
def stop_job(job_name: str):
    """Stop a job.

    Returns:
        None.
    """

    redis_controller.remove_pending_job_ticket(
        cluster_name=local_cluster_details["name"],
        job_name=job_name
    )
    redis_controller.push_killed_job_ticket(
        cluster_name=local_cluster_details["name"],
        job_name=job_name
    )
    return {}


@blueprint.route(f"{URL_PREFIX}:clean", methods=["POST"])
def clean_jobs():
    """Clean all jobs in the cluster.

    Returns:
        None.
    """

    # Get params
    master_api_server_port = local_master_details["api_server"]["port"]

    # Delete all job related resources.
    redis_controller.delete_pending_jobs_queue(cluster_name=local_cluster_details["name"])
    redis_controller.delete_killed_jobs_queue(cluster_name=local_cluster_details["name"])
    name_to_node_details = redis_controller.get_name_to_node_details(
        cluster_name=local_cluster_details["name"]
    )
    for _, node_details in name_to_node_details.items():
        node_hostname = node_details["hostname"]
        for container_name, container_details in node_details["containers"].items():
            requests.delete(url=f"http://{node_hostname}:{master_api_server_port}/containers/{container_name}")
    return {}
