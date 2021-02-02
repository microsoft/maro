from flask import Blueprint

from ..jwt_wrapper import check_jwt_validity
from ..objects import redis_controller

# Flask related.

blueprint = Blueprint(name="visible", import_name=__name__)
URL_PREFIX = "/v1/visible"


# Api functions.

@blueprint.route(f"{URL_PREFIX}/static", methods=["GET"])
@check_jwt_validity
def get_static_resource():
    """Get the job with job_name.

    Returns:
        None.
    """

    name_to_node_resources = redis_controller.get_name_to_node_resources()
    return name_to_node_resources


@blueprint.route(f"{URL_PREFIX}/dynamic/<previous_length>", methods=["GET"])
@check_jwt_validity
def get_dynamic_resource(previous_length: str):
    """Get the job with job_name.

    Returns:
        None.
    """

    name_to_node_usage = redis_controller.get_resource_usage(
        previous_length=int(previous_length)
    )
    return name_to_node_usage
