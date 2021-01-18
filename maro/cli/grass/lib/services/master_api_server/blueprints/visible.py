import requests
from flask import Blueprint, jsonify, request

from ..objects import redis_controller, local_cluster_details, local_master_details


# Flask related.

blueprint = Blueprint(name="visible", import_name=__name__)
URL_PREFIX = "/v1/visible"


# Api functions.

@blueprint.route(f"{URL_PREFIX}/static", methods=["GET"])
def get_job(job_name: str):
    """Get the job with job_name.

    Returns:
        None.
    """

    node_details = redis_controller.get_node_details(
        cluster_name=local_cluster_details["name"],
        job_name=job_name
    )
    return node_details


@blueprint.route(f"{URL_PREFIX}/dynamic", methods=["GET"])
def get_job(job_name: str):
    """Get the job with job_name.

    Returns:
        None.
    """

    node_details = redis_controller.get_resource_usage(
        cluster_name=local_cluster_details["name"],
        node_name=node_name,
        job_name=job_name
    )
    return node_details
