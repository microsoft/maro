# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Blueprint

from ..jwt_wrapper import check_jwt_validity
from ..objects import redis_controller

# Flask related.

blueprint = Blueprint(name="image_files", import_name=__name__)
URL_PREFIX = "/v1/imageFiles"


# Api functions.


@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
@check_jwt_validity
def list_image_files():
    """List the image files in the cluster.

    Returns:
        None.
    """

    master_details = redis_controller.get_master_details()
    return list(master_details["image_files"].values())


@blueprint.route(f"{URL_PREFIX}/<image_file_name>", methods=["GET"])
@check_jwt_validity
def get_image_file(image_file_name: str):
    """Get the image file.

    Returns:
        None.
    """

    master_details = redis_controller.get_master_details()
    return master_details["image_files"].get(image_file_name, {})


@blueprint.route(f"{URL_PREFIX}", methods=["POST"])
@check_jwt_validity
def create_image_file(**kwargs):
    """Create a image file.

    Returns:
        None.
    """

    image_file_details = kwargs["json_dict"]
    with redis_controller.lock(name="lock:master_details"):
        master_details = redis_controller.get_master_details()
        master_details["image_files"][image_file_details["name"]] = image_file_details
        redis_controller.set_master_details(master_details=master_details)
    return {}
