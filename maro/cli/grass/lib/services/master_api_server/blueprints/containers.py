# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Blueprint

from ..jwt_wrapper import check_jwt_validity
from ..objects import redis_controller

# Flask related.

blueprint = Blueprint(name="containers", import_name=__name__)
URL_PREFIX = "/v1/containers"


# Api functions.

@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
@check_jwt_validity
def list_containers():
    """List the jobs in the cluster.

    Returns:
        None.
    """

    name_to_container_details = redis_controller.get_name_to_container_details()
    return list(name_to_container_details.values())
