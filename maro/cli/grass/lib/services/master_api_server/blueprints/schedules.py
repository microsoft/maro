# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import copy

from flask import Blueprint, jsonify, request

from ..objects import redis_controller, service_config

# Flask related.

blueprint = Blueprint(name="schedules", import_name=__name__)
URL_PREFIX = "/schedules"


# Api functions.

@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
def list_schedules():
    """List the schedules in the cluster.

    Returns:
        None.
    """

    name_to_schedule_details = redis_controller.get_name_to_schedule_details(
        cluster_name=service_config["cluster_name"]
    )
    return jsonify(list(name_to_schedule_details.values()))


@blueprint.route(f"{URL_PREFIX}/<schedule_name>", methods=["GET"])
def get_schedule(schedule_name: str):
    """Get the schedule with schedule_name.

    Returns:
        None.
    """

    schedule_details = redis_controller.get_schedule_details(
        cluster_name=service_config["cluster_name"],
        schedule_name=schedule_name
    )
    return schedule_details


@blueprint.route(f"{URL_PREFIX}", methods=["POST"])
def create_schedule():
    """Create a schedule.

    Returns:
        None.
    """

    schedule_details = request.json
    redis_controller.set_schedule_details(
        cluster_name=service_config["cluster_name"],
        schedule_name=schedule_details["name"],
        schedule_details=schedule_details
    )

    # Build individual jobs
    for job_name in schedule_details["job_names"]:
        redis_controller.set_job_details(
            cluster_name=service_config["cluster_name"],
            job_name=job_name,
            job_details=_build_job_details(schedule_details=schedule_details, job_name=job_name)
        )
        redis_controller.push_pending_job_ticket(
            cluster_name=service_config["cluster_name"],
            job_name=job_name
        )
    return {}


@blueprint.route(f"{URL_PREFIX}/<schedule_name>", methods=["DELETE"])
def delete_schedule(schedule_name: str):
    """Delete a schedule.

    Returns:
        None.
    """

    schedule_details = redis_controller.get_schedule_details(
        cluster_name=service_config["cluster_name"],
        schedule_name=schedule_name
    )
    for job_name in schedule_details["job_names"]:
        redis_controller.remove_pending_job_ticket(
            cluster_name=service_config["cluster_name"],
            job_name=job_name
        )
        redis_controller.push_killed_job_ticket(
            cluster_name=service_config["cluster_name"],
            job_name=job_name
        )
    redis_controller.delete_schedule_details(
        cluster_name=service_config["cluster_name"],
        schedule_name=schedule_name
    )
    return {}


@blueprint.route(f"{URL_PREFIX}/<schedule_name>:stop", methods=["POST"])
def stop_schedule(schedule_name: str):
    """Stop a schedule.

    Returns:
        None.
    """

    schedule_details = redis_controller.get_schedule_details(
        cluster_name=service_config["cluster_name"],
        schedule_name=schedule_name
    )
    for job_name in schedule_details["job_names"]:
        redis_controller.remove_pending_job_ticket(
            cluster_name=service_config["cluster_name"],
            job_name=job_name
        )
        redis_controller.push_killed_job_ticket(
            cluster_name=service_config["cluster_name"],
            job_name=job_name
        )
    return {}


def _build_job_details(schedule_details: dict, job_name: str) -> dict:
    schedule_name = schedule_details["name"]

    job_details = copy.deepcopy(schedule_details)
    job_details["name"] = job_name
    job_details["tags"] = {
        "schedule": schedule_name
    }
    job_details.pop("job_names")

    return job_details
