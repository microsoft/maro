# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from flask import Blueprint, jsonify, request, abort

from ..objects import redis_controller, local_cluster_details
from ...utils.connection_tester import ConnectionTester
from ...utils.exception import ConnectionFailed
from ...utils.name_creator import NameCreator
from ...utils.params import NodeStatus, Paths
from ...utils.subprocess import SubProcess

# Flask related.

blueprint = Blueprint(name="nodes", import_name=__name__)
URL_PREFIX = "/v1/nodes"


# Api functions.

@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
def list_nodes():
    """List the nodes in the cluster.

    Returns:
        None.
    """

    name_to_node_details = redis_controller.get_name_to_node_details(cluster_name=local_cluster_details["name"])
    return jsonify(list(name_to_node_details.values()))


@blueprint.route(f"{URL_PREFIX}/<node_name>", methods=["GET"])
def get_node(node_name: str):
    """Get the node with node_name.

    Returns:
        None.
    """

    node_details = redis_controller.get_node_details(
        cluster_name=local_cluster_details["name"],
        node_name=node_name
    )
    return node_details


@blueprint.route(f"{URL_PREFIX}", methods=["POST"])
def create_node():
    """Create a node.

    Returns:
        None.
    """

    node_details = request.json

    # Init runtime params.
    if "name" not in node_details and "id" not in node_details:
        node_name = NameCreator.create_node_name()
        node_details["name"] = node_name
        node_details["id"] = node_name
    node_details["image_files"] = {}
    node_details["containers"] = {}
    node_details["state"] = {
        "status": NodeStatus.PENDING
    }

    node_name = node_details["name"]
    with redis_controller.lock(f"lock:name_to_node_details:{node_name}"):
        redis_controller.set_node_details(
            cluster_name=local_cluster_details["name"],
            node_name=node_name,
            node_details=node_details
        )
    return node_details


@blueprint.route(f"{URL_PREFIX}/<node_name>", methods=["DELETE"])
def delete_node(node_name: str):
    """Delete a node.

    Returns:
        None.
    """

    # Get node_details.
    node_details = redis_controller.get_node_details(
        cluster_name=local_cluster_details["name"],
        node_name=node_name
    )

    # leave the cluster
    command = (
        f"ssh -o StrictHostKeyChecking=no "
        f"-i {Paths.MARO_LOCAL}/cluster/{local_cluster_details['name']}/id_rsa_master "
        f"-p {node_details['ssh']['port']} "
        f"{node_details['username']}@{node_details['hostname']} "
        f"'python3 {Paths.MARO_LOCAL}/scripts/leave_cluster.py'"
    )
    SubProcess.run(command=command)

    # Delete node_details at the end.
    redis_controller.delete_node_details(
        cluster_name=local_cluster_details["name"],
        node_name=node_name
    )

    return node_details


@blueprint.route(f"{URL_PREFIX}/<node_name>:start", methods=["POST"])
def start_node(node_name: str):
    """Start a node.

    Returns:
        None.
    """

    node_details = redis_controller.get_node_details(
        cluster_name=local_cluster_details["name"],
        node_name=node_name
    )
    # Make sure the node is able to connect
    try:
        ConnectionTester.retry_connection(
            node_username=node_details["username"],
            node_hostname=node_details["hostname"],
            node_ssh_port=node_details["ssh"]["port"],
            cluster_name=local_cluster_details["name"]
        )
    except ConnectionFailed:
        abort(400)

    command = (
        f"ssh -o StrictHostKeyChecking=no "
        f"-i {Paths.MARO_LOCAL}/cluster/{local_cluster_details['name']}/id_rsa_master "
        f"-p {node_details['ssh']['port']} "
        f"{node_details['username']}@{node_details['hostname']} "
        f"'cd {Paths.MARO_SHARED}/lib/grass; python3 -m scripts.node.start_node_agent_service'"
    )
    _ = SubProcess.run(command)
    command = (
        f"ssh -o StrictHostKeyChecking=no "
        f"-i {Paths.MARO_LOCAL}/cluster/{local_cluster_details['name']}/id_rsa_master "
        f"-p {node_details['ssh']['port']} "
        f"{node_details['username']}@{node_details['hostname']} "
        f"'cd {Paths.MARO_SHARED}/lib/grass; python3 -m scripts.node.start_node_api_server_service'"
    )
    _ = SubProcess.run(command)

    return {}


@blueprint.route(f"{URL_PREFIX}/<node_name>:stop", methods=["POST"])
def stop_node(node_name: str):
    """Stop a node.

    Returns:
        None.
    """

    node_details = redis_controller.get_node_details(
        cluster_name=local_cluster_details["name"],
        node_name=node_name
    )
    # Make sure the node is able to connect
    try:
        ConnectionTester.retry_connection(
            node_username=node_details["username"],
            node_hostname=node_details["hostname"],
            node_ssh_port=node_details["ssh"]["port"],
            cluster_name=local_cluster_details["name"]
        )
    except ConnectionFailed:
        abort(400)

    command = (
        f"ssh -o StrictHostKeyChecking=no "
        f"-i {Paths.MARO_LOCAL}/cluster/{local_cluster_details['name']}/id_rsa_master "
        f"-p {node_details['ssh']['port']} "
        f"{node_details['username']}@{node_details['hostname']} "
        f"'cd {Paths.MARO_SHARED}/lib/grass; python3 -m scripts.node.stop_node_api_server_service'"
    )
    _ = SubProcess.run(command)
    command = (
        f"ssh -o StrictHostKeyChecking=no "
        f"-i {Paths.MARO_LOCAL}/cluster/{local_cluster_details['name']}/id_rsa_master "
        f"-p {node_details['ssh']['port']} "
        f"{node_details['username']}@{node_details['hostname']} "
        f"'cd {Paths.MARO_SHARED}/lib/grass; python3 -m scripts.node.stop_node_agent_service'"
    )
    _ = SubProcess.run(command)

    return {}
