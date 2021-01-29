# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import stat

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from flask import Blueprint

from ...master_api_server.jwt_wrapper import check_jwt_validity
from ...master_api_server.objects import local_cluster_details, redis_controller
from ...utils.params import Paths

# Flask related.

blueprint = Blueprint(name="master", import_name=__name__)
URL_PREFIX = "/v1/master"


# Api functions.

@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
@check_jwt_validity
def get_master():
    """Get master.

    Returns:
        None.
    """

    master_details = redis_controller.get_master_details()
    return master_details


@blueprint.route(f"{URL_PREFIX}", methods=["POST"])
@check_jwt_validity
def create_master(**kwargs):
    """Create master.

    Returns:
        None.
    """

    master_details = kwargs["json_dict"]

    # Create rsa key-pair for master-node communication
    master_node_key_pair = generate_rsa_openssh_key_pair()
    save_master_key(private_key=master_node_key_pair["private_key"])

    # Init runtime params.
    master_details["image_files"] = {}
    master_details["ssh"]["public_key"] = master_node_key_pair["public_key"]

    redis_controller.set_master_details(master_details=master_details)

    return master_details


@blueprint.route(f"{URL_PREFIX}", methods=["DELETE"])
@check_jwt_validity
def delete_master():
    """Delete master.

    Returns:
        None.
    """

    redis_controller.delete_master_details()
    return {}


def save_master_key(private_key: str) -> None:
    cluster_name = local_cluster_details["name"]
    os.makedirs(name=f"{Paths.ABS_MARO_LOCAL}/cluster/{cluster_name}", exist_ok=True)
    with open(file=f"{Paths.ABS_MARO_LOCAL}/cluster/{cluster_name}/master_to_node_openssh_private_key", mode="w") as fw:
        fw.write(private_key)
    os.chmod(
        path=f"{Paths.ABS_MARO_LOCAL}/cluster/{cluster_name}/master_to_node_openssh_private_key",
        mode=stat.S_IRWXU
    )


def generate_rsa_openssh_key_pair() -> dict:
    rsa_key = rsa.generate_private_key(
        backend=default_backend(),
        public_exponent=65537,
        key_size=2048
    )

    # Format and encoding are diff from OpenSSH
    private_key = rsa_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_key = rsa_key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH
    )
    return {
        "public_key": public_key.decode("utf-8"),
        "private_key": private_key.decode("utf-8")
    }
