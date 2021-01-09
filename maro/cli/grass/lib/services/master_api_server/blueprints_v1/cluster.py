# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Blueprint

from ..objects import redis_controller, service_config
from ...utils.details_reader import DetailsReader

# Flask related.

blueprint = Blueprint(name="cluster", import_name=__name__)
URL_PREFIX = "/v1/cluster"


# Api functions.

@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
def get_cluster():
    """Get cluster.

    Returns:
        None.
    """

    cluster_details = DetailsReader.load_cluster_details(cluster_name=service_config["cluster_name"])
    cluster_details["master"] = redis_controller.get_master_details(cluster_name=service_config["cluster_name"])
    return cluster_details
