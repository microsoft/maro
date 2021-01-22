# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import base64
import functools
import json
import logging
import os

import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from flask import Response, abort, request

from .objects import redis_controller

logger = logging.getLogger(name=__name__)


def check_jwt_validity(func):
    """Check JWT validity and do data decryption before getting into the actual logistic.

    Args:
        func:

    Returns:
        None.
    """

    @functools.wraps(func)
    def with_checker(*args, **kwargs):
        # Get jwt_token and its payload
        authorization = request.headers.get("Authorization", None)
        if not authorization:
            abort(401)
        jwt_token = authorization.split()[1]
        payload = jwt.decode(jwt=jwt_token, options={"verify_signature": False})

        # Get user_details
        user_details = redis_controller.get_user_details(user_id=payload["user_id"])

        # Get decrypted_bytes
        if request.data != b'':
            decrypted_bytes = _get_decrypted_bytes(
                payload=payload,
                encrypted_bytes=request.data,
                user_details=user_details
            )
            kwargs["json_dict"] = json.loads(decrypted_bytes.decode("utf-8"))

        # Check JWT token validity
        try:
            jwt.decode(jwt=jwt_token, key=user_details["dev_to_master_signing_public_key"], algorithms=["RS512"])
        except jwt.ExpiredSignatureError:
            abort(401)
        except jwt.InvalidTokenError:
            abort(401)

        # Do actual HTTP call
        return_json = func(*args, **kwargs)

        return build_response(return_json=return_json, user_details=user_details)

    return with_checker


def _get_encrypted_bytes(json_dict: dict, aes_key: bytes, aes_ctr_nonce: bytes) -> bytes:
    cipher = Cipher(
        algorithm=algorithms.AES(key=aes_key),
        mode=modes.CTR(nonce=aes_ctr_nonce)
    )
    encryptor = cipher.encryptor()
    return_bytes = encryptor.update(json.dumps(json_dict).encode("utf-8")) + encryptor.finalize()
    return return_bytes


def _get_decrypted_bytes(payload: dict, encrypted_bytes: bytes, user_details: dict) -> bytes:
    # Decrypted aes_key and aes_ctr_nonce
    dev_to_master_encryption_private_key_obj = serialization.load_pem_private_key(
        data=user_details["dev_to_master_encryption_private_key"].encode("utf-8"),
        password=None
    )
    aes_key = dev_to_master_encryption_private_key_obj.decrypt(
        ciphertext=base64.b64decode(payload["aes_key"].encode("ascii")),
        padding=_get_asymmetric_padding()
    )
    aes_ctr_nonce = dev_to_master_encryption_private_key_obj.decrypt(
        ciphertext=base64.b64decode(payload["aes_ctr_nonce"].encode("ascii")),
        padding=_get_asymmetric_padding()
    )

    # Return decrypted_bytes
    cipher = Cipher(
        algorithm=algorithms.AES(key=aes_key),
        mode=modes.CTR(nonce=aes_ctr_nonce)
    )
    decryptor = cipher.decryptor()
    return decryptor.update(encrypted_bytes) + decryptor.finalize()


def build_response(return_json: dict, user_details: dict) -> Response:
    # Build random aes related params
    aes_key = os.urandom(32)
    aes_ctr_nonce = os.urandom(16)

    # Get encrypted_bytes
    encrypted_bytes = _get_encrypted_bytes(
        json_dict=return_json,
        aes_key=aes_key,
        aes_ctr_nonce=aes_ctr_nonce
    )

    # Encrypt aes_key and aes_ctr_nonce with rsa_key_pair
    master_to_dev_encryption_public_key_obj = serialization.load_pem_public_key(
        data=user_details["master_to_dev_encryption_public_key"].encode("utf-8")
    )

    # Build jwt_token
    jwt_token = jwt.encode(
        payload={
            "aes_key": base64.b64encode(
                master_to_dev_encryption_public_key_obj.encrypt(
                    plaintext=aes_key,
                    padding=_get_asymmetric_padding()
                )
            ).decode("ascii"),
            "aes_ctr_nonce": base64.b64encode(
                master_to_dev_encryption_public_key_obj.encrypt(
                    plaintext=aes_ctr_nonce,
                    padding=_get_asymmetric_padding()
                )
            ).decode("ascii")
        },
        key=user_details["master_to_dev_signing_private_key"],
        algorithm="RS512"
    )

    # Build response
    response = Response(response=encrypted_bytes)
    response.headers["Authorization"] = f"Bearer {jwt_token}"
    return response


def _get_asymmetric_padding():
    return padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
