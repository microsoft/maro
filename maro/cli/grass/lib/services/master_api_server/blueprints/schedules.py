# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import copy

from flask import Blueprint, jsonify, request

from ..objects import redis_controller, local_cluster_details
from ...utils.name_creator import NameCreator

# Flask related.

blueprint = Blueprint(name="schedules", import_name=__name__)
URL_PREFIX = "/v1/schedules"


# Api functions.

@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
def list_schedules():
    """List the schedules in the cluster.

    Returns:
        None.
    """

    name_to_schedule_details = redis_controller.get_name_to_schedule_details(
        cluster_name=local_cluster_details["name"]
    )
    return jsonify(list(name_to_schedule_details.values()))


@blueprint.route(f"{URL_PREFIX}/<schedule_name>", methods=["GET"])
def get_schedule(schedule_name: str):
    """Get the schedule with schedule_name.

    Returns:
        None.
    """

    schedule_details = redis_controller.get_schedule_details(
        cluster_name=local_cluster_details["name"],
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
        cluster_name=local_cluster_details["name"],
        schedule_name=schedule_details["name"],
        schedule_details=schedule_details
    )

    # Build individual jobs
    for job_name in schedule_details["job_names"]:
        redis_controller.set_job_details(
            cluster_name=local_cluster_details["name"],
            job_name=job_name,
            job_details=_build_job_details(schedule_details=schedule_details, job_name=job_name)
        )
        redis_controller.push_pending_job_ticket(
            cluster_name=local_cluster_details["name"],
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
        cluster_name=local_cluster_details["name"],
        schedule_name=schedule_name
    )
    for job_name in schedule_details["job_names"]:
        redis_controller.remove_pending_job_ticket(
            cluster_name=local_cluster_details["name"],
            job_name=job_name
        )
        redis_controller.push_killed_job_ticket(
            cluster_name=local_cluster_details["name"],
            job_name=job_name
        )
    redis_controller.delete_schedule_details(
        cluster_name=local_cluster_details["name"],
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
        cluster_name=local_cluster_details["name"],
        schedule_name=schedule_name
    )
    for job_name in schedule_details["job_names"]:
        # FIXME: use schedule id to check
        redis_controller.remove_pending_job_ticket(
            cluster_name=local_cluster_details["name"],
            job_name=job_name
        )
        redis_controller.push_killed_job_ticket(
            cluster_name=local_cluster_details["name"],
            job_name=job_name
        )
    return {}


def _build_job_details(schedule_details: dict, job_name: str) -> dict:
    job_details = copy.deepcopy(schedule_details)

    # Convert schedule_details to job_details
    job_details["name"] = job_name
    job_details["tags"] = {
        "schedule_name": schedule_details["name"],
        "schedule_id": schedule_details["id"]
    }
    job_details.pop("job_names")

    # Init runtime params
    job_details["id"] = NameCreator.create_job_id()
    job_details["containers"] = {}
    for _, component_details in job_details["components"].items():
        component_details["id"] = NameCreator.create_component_id()

    return job_details
