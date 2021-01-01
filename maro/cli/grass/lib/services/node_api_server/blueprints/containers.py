# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Blueprint, request, abort

from ...utils.docker_controller import DockerController
from ...utils.exception import CommandExecutionError

# Flask related.

blueprint = Blueprint(name="container", import_name=__name__, url_prefix="/container")


# Api functions.

@blueprint.route("", methods=["POST"])
def create():
    """Create a container, aka 'docker run'.

    Returns:
        None.
    """

    try:
        create_config = request.json
        return DockerController.create_container_with_config(create_config=create_config)
    except CommandExecutionError:
        abort(400)


@blueprint.route("/<container_name>", methods=["DELETE"])
def delete(container_name: str):
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


@blueprint.route("/<container_name>:stop", methods=["POST"])
def stop(container_name: str):
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
