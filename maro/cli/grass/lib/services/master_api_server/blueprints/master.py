# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import stat

from cryptography.hazmat.backends import default_backend as crypto_default_backend
from cryptography.hazmat.primitives import serialization as crypto_serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from flask import Blueprint, request

from ..objects import redis_controller, local_cluster_details
from ...utils.params import Paths

# Flask related.

blueprint = Blueprint(name="master", import_name=__name__)
URL_PREFIX = "/v1/master"


# Api functions.

@blueprint.route(f"{URL_PREFIX}", methods=["GET"])
def get_master():
    """Get master.

    Returns:
        None.
    """

    master_details = redis_controller.get_master_details(cluster_name=local_cluster_details["name"])
    return master_details


@blueprint.route(f"{URL_PREFIX}", methods=["POST"])
def create_master():
    """Create master.

    Returns:
        None.
    """

    master_details = request.json

    # Create ssh-key for master-node communication
    public_key = generate_master_key()

    # Init runtime params.
    master_details["image_files"] = {}
    master_details["ssh"]["public_key"] = public_key

    redis_controller.set_master_details(
        cluster_name=local_cluster_details["name"],
        master_details=master_details
    )

    return master_details


@blueprint.route(f"{URL_PREFIX}", methods=["DELETE"])
def delete_master():
    """Delete master.

    Returns:
        None.
    """

    redis_controller.delete_master_details(cluster_name=local_cluster_details["name"])
    return {}


def generate_master_key() -> str:
    key = rsa.generate_private_key(
        backend=crypto_default_backend(),
        public_exponent=65537,
        key_size=2048
    )
    private_key = key.private_bytes(
        encoding=crypto_serialization.Encoding.PEM,
        format=crypto_serialization.PrivateFormat.PKCS8,
        encryption_algorithm=crypto_serialization.NoEncryption())
    public_key = key.public_key().public_bytes(
        encoding=crypto_serialization.Encoding.OpenSSH,
        format=crypto_serialization.PublicFormat.OpenSSH
    )

    cluster_name = local_cluster_details["name"]
    os.makedirs(name=f"{Paths.ABS_MARO_LOCAL}/cluster/{cluster_name}", exist_ok=True)
    with open(
        file=f"{Paths.ABS_MARO_LOCAL}/cluster/{cluster_name}/id_rsa_master",
        mode="wb"
    ) as fw:
        fw.write(private_key)
    os.chmod(
        path=f"{Paths.ABS_MARO_LOCAL}/cluster/{cluster_name}/id_rsa_master",
        mode=stat.S_IRWXU
    )
    return public_key.decode("utf-8")
