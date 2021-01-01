# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Blueprint, jsonify, request

from ..objects import redis_controller, service_config

# Flask related.

blueprint = Blueprint(name="jobs", import_name=__name__, url_prefix="/jobs")


# Api functions.

@blueprint.route("", methods=["GET"])
def list_jobs():
    """List the jobs in the cluster.

    Returns:
        None.
    """

    name_to_job_details = redis_controller.get_name_to_job_details(cluster_name=service_config["cluster_name"])
    return jsonify(list(name_to_job_details.values()))


@blueprint.route("/<job_name>", methods=["GET"])
def get_job(job_name: str):
    """Get the job with job_name.

    Returns:
        None.
    """

    job_details = redis_controller.get_job_details(
        cluster_name=service_config["cluster_name"],
        job_name=job_name
    )
    return job_details


@blueprint.route("", methods=["POST"])
def create_job():
    """Create a job.

    Returns:
        None.
    """

    job_details = request.json
    redis_controller.set_job_details(
        cluster_name=service_config["cluster_name"],
        job_name=job_details["name"],
        job_details=job_details
    )
    redis_controller.create_pending_job_ticket(
        cluster_name=service_config["cluster_name"],
        job_name=job_details["name"]
    )
    return {}


@blueprint.route("/<job_name>", methods=["DELETE"])
def delete_job(job_name: str):
    """Delete a job.

    Returns:
        None.
    """

    redis_controller.remove_pending_job_ticket(
        cluster_name=service_config["cluster_name"],
        job_name=job_name
    )
    redis_controller.create_killed_job_ticket(
        cluster_name=service_config["cluster_name"],
        job_name=job_name
    )
    return {}
