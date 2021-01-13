# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Blueprint, jsonify

from ..objects import redis_controller, local_cluster_details

# Flask related.

blueprint = Blueprint(name="containers", import_name=__name__)
URL_PREFIX = "/v1/containers"


# Api functions.

@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
def list_containers():
    """List the jobs in the cluster.

    Returns:
        None.
    """

    name_to_container_details = redis_controller.get_name_to_container_details(
        cluster_name=local_cluster_details["name"]
    )
    return jsonify(list(name_to_container_details.values()))
