# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Blueprint, jsonify, request

from ..objects import redis_controller, local_cluster_details

# Flask related.

blueprint = Blueprint(name="image_files", import_name=__name__)
URL_PREFIX = "/v1/imageFiles"


# Api functions.


@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
def list_image_files():
    """List the image files in the cluster.

    Returns:
        None.
    """

    master_details = redis_controller.get_master_details(cluster_name=local_cluster_details["name"])
    return jsonify(list(master_details["image_files"].values()))


@blueprint.route(f"{URL_PREFIX}/<image_file_name>", methods=["GET"])
def get_image_file(image_file_name: str):
    """Get the image file.

    Returns:
        None.
    """

    master_details = redis_controller.get_master_details(cluster_name=local_cluster_details["name"])
    return master_details["image_files"].get(image_file_name, {})


@blueprint.route(f"{URL_PREFIX}", methods=["POST"])
def create_image_file():
    """Create a image file.

    Returns:
        None.
    """

    image_file_details = request.json
    with redis_controller.lock(name="lock:master_details"):
        master_details = redis_controller.get_master_details(cluster_name=local_cluster_details["name"])
        master_details["image_files"][image_file_details["name"]] = image_file_details
        redis_controller.set_master_details(
            cluster_name=local_cluster_details["name"],
            master_details=master_details
        )
    return {}
