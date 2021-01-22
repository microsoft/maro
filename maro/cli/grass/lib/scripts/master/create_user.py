# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""Create MARO User.

MARO User is a MARO resource for local-master authorization.
Four pairs of public-private keys will be created here, some of the keys will be kept in the master,
the others will be sent back to local.
Then, local will use these keys to encrypt token and sign signature when communicating to Master API Server.
The communication is encrypted with RSA+AES in a hybrid mode, follow the JWT standard.

The script will do the following jobs in this VM:
- Create four pairs of keys, save some of them to Redis, and print some of them to the command line.
"""

import argparse
import json
import sys

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from ..utils.details_reader import DetailsReader
from ..utils.redis_controller import RedisController


class UserCreator:
    def __init__(self, local_cluster_details: dict):
        self._local_cluster_details = local_cluster_details
        self._redis_controller = RedisController(
            host="localhost",
            port=self._local_cluster_details["master"]["redis"]["port"]
        )

    @staticmethod
    def _generate_rsa_key_pair() -> dict:
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
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return {
            "public_key": public_key.decode("utf-8"),
            "private_key": private_key.decode("utf-8")
        }

    def create_user(self, user_id: str, user_role: str) -> None:
        dev_to_master_encryption_key_pair = self._generate_rsa_key_pair()
        master_to_dev_encryption_key_pair = self._generate_rsa_key_pair()
        dev_to_master_signing_key_pair = self._generate_rsa_key_pair()
        master_to_dev_signing_key_pair = self._generate_rsa_key_pair()

        # Set user details.
        self._redis_controller.set_user_details(
            user_id=user_id,
            user_details={
                "id": user_id,
                "role": user_role,
                "master_to_dev_encryption_public_key": master_to_dev_encryption_key_pair["public_key"],
                "dev_to_master_encryption_private_key": dev_to_master_encryption_key_pair["private_key"],
                "dev_to_master_signing_public_key": dev_to_master_signing_key_pair["public_key"],
                "master_to_dev_signing_private_key": master_to_dev_signing_key_pair["private_key"]
            }
        )

        # Write private key to console.
        sys.stdout.write(
            json.dumps(
                {
                    "id": user_id,
                    "user_role": user_role,
                    "master_to_dev_encryption_private_key": master_to_dev_encryption_key_pair["private_key"],
                    "dev_to_master_encryption_public_key": dev_to_master_encryption_key_pair["public_key"],
                    "dev_to_master_signing_private_key": dev_to_master_signing_key_pair["private_key"],
                    "master_to_dev_signing_public_key": master_to_dev_signing_key_pair["public_key"]
                }
            )
        )


if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id")
    parser.add_argument("user_role")
    args = parser.parse_args()

    # Start initializing
    user_creator = UserCreator(local_cluster_details=DetailsReader.load_local_cluster_details())
    user_creator.create_user(user_id=args.user_id, user_role=args.user_role)
