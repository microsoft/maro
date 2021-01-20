# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Blueprint, abort, request

from ...utils.docker_controller import DockerController
from ...utils.exception import CommandExecutionError

# Flask related.

blueprint = Blueprint(name="container", import_name=__name__)
URL_PREFIX = "/v1/containers"


# Api functions.

@blueprint.route(f"{URL_PREFIX}", methods=["POST"])
def create_container():
    """Create a container, aka 'docker run'.

    Returns:
        None.
    """

    try:
        create_config = request.json
        return DockerController.create_container_with_config(create_config=create_config)
    except CommandExecutionError:
        abort(400)


@blueprint.route(f"{URL_PREFIX}/<container_name>", methods=["DELETE"])
def delete_container(container_name: str):
    """Delete a container, aka 'docker rm'.

    Args:
        container_name (str): Name of the container.

    Returns:
        None.
    """

    try:
        DockerController.remove_container(container_name=container_name)
        return {}
    except CommandExecutionError:
        abort(400)


@blueprint.route(f"{URL_PREFIX}/<container_name>:stop", methods=["POST"])
def stop_container(container_name: str):
    """Stop a container, aka 'docker stop'.

    Args:
        container_name (str): Name of the container.

    Returns:
        None.
    """

    try:
        DockerController.stop_container(container_name=container_name)
        return {}
    except CommandExecutionError:
        abort(400)
