# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Blueprint

from ..jwt_wrapper import check_jwt_validity
from ..objects import redis_controller

# Flask related.

blueprint = Blueprint(name="cluster", import_name=__name__)
URL_PREFIX = "/v1/cluster"


# Api functions.

@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
@check_jwt_validity
def get_cluster():
    """Get cluster.

    Returns:
        None.
    """

    cluster_details = redis_controller.get_cluster_details()
    cluster_details["master"] = redis_controller.get_master_details()
    return cluster_details


@blueprint.route(f"{URL_PREFIX}", methods=["POST"])
@check_jwt_validity
def create_cluster(**kwargs):
    """Create cluster.

    Returns:
        None.
    """

    cluster_details = kwargs["json_dict"]

    redis_controller.set_cluster_details(cluster_details=cluster_details)
    return cluster_details
