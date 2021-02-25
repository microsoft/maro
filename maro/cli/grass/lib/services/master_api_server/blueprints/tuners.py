# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Blueprint

from ..jwt_wrapper import check_jwt_validity
from ..objects import redis_controller

# Flask related.

blueprint = Blueprint(name="tuners", import_name=__name__)
URL_PREFIX = "/v1/tuners"


@blueprint.route(f"{URL_PREFIX}", methods=["POST"])
@check_jwt_validity
def create_tuner(**kwargs):
    """Create a tuner.
    Returns:
        None.
    """

    tuner_details = kwargs["json_dict"]
    redis_controller.set_job_details(
        job_name=tuner_details["name"],
        job_details=tuner_details
    )
    redis_controller.push_pending_job_ticket(
        job_name=tuner_details["name"]
    )
    return {}


@blueprint.route(f"{URL_PREFIX}/<tuner_name>:stop", methods=["POST"])
@check_jwt_validity
def stop_tuner(tuner_name: str):
    """Stop a tuner.
    Returns:
        None.
    """

    tuner_details = redis_controller.get_job_details(job_name=tuner_name)
    redis_controller.remove_pending_job_ticket(job_name=tuner_details["name"])
    redis_controller.push_killed_job_ticket(job_name=tuner_details["name"])
    for job_name in tuner_details["job_names"]:
        redis_controller.remove_pending_job_ticket(job_name=job_name)
        redis_controller.push_killed_job_ticket(job_name=job_name)
    return {}
