# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import base64
import datetime
import json
import os

import jwt
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class EncryptedRequests:
    """Wrapper class for requests with encryption/decryption integrated.
    """

    def __init__(
        self,
        user_id: str,
        master_to_dev_encryption_private_key: str,
        dev_to_master_encryption_public_key: str,
        dev_to_master_signing_private_key: str
    ):
        self._user_id = user_id
        self._dev_to_master_signing_private_key = dev_to_master_signing_private_key
        self._master_to_dev_encryption_private_key_obj = serialization.load_pem_private_key(
            master_to_dev_encryption_private_key.encode("utf-8"),
            password=None
        )
        self._dev_to_master_encryption_public_key_obj = serialization.load_pem_public_key(
            dev_to_master_encryption_public_key.encode("utf-8")
        )

    def get(self, url: str):
        # Build random aes related params
        aes_key = os.urandom(32)
        aes_ctr_nonce = os.urandom(16)

        # Get response.
        response = requests.get(
            url=url,
            headers=self._get_new_headers(
                aes_key=aes_key,
                aes_ctr_nonce=aes_ctr_nonce
            )
        )

        # Parse response
        decrypted_bytes = self._get_decrypted_bytes(response=response)
        return json.loads(decrypted_bytes.decode("utf-8"))

    def post(self, url: str, json_dict: dict = None):
        # Build random aes related params
        aes_key = os.urandom(32)
        aes_ctr_nonce = os.urandom(16)

        # Get response.
        response = requests.post(
            url=url,
            headers=self._get_new_headers(
                aes_key=aes_key,
                aes_ctr_nonce=aes_ctr_nonce
            ),
            data=None if json_dict is None else self._get_encrypted_bytes(
                json_dict=json_dict,
                aes_key=aes_key,
                aes_ctr_nonce=aes_ctr_nonce
            )
        )

        # Parse response
        decrypted_bytes = self._get_decrypted_bytes(response=response)
        return json.loads(decrypted_bytes.decode("utf-8"))

    def delete(self, url: str, json_dict: dict = None):
        # Build random aes related params
        aes_key = os.urandom(32)
        aes_ctr_nonce = os.urandom(16)

        # Get response.
        response = requests.delete(
            url=url,
            headers=self._get_new_headers(
                aes_key=aes_key,
                aes_ctr_nonce=aes_ctr_nonce
            ),
            data=None if json_dict is None else self._get_encrypted_bytes(
                json_dict=json_dict,
                aes_key=aes_key,
                aes_ctr_nonce=aes_ctr_nonce
            )
        )

        # Parse response
        decrypted_bytes = self._get_decrypted_bytes(response=response)
        return json.loads(decrypted_bytes.decode("utf-8"))

    # Utils

    @staticmethod
    def _get_encrypted_bytes(json_dict: dict, aes_key: bytes, aes_ctr_nonce: bytes) -> bytes:
        cipher = Cipher(
            algorithm=algorithms.AES(key=aes_key),
            mode=modes.CTR(nonce=aes_ctr_nonce)
        )
        encryptor = cipher.encryptor()
        return_bytes = encryptor.update(json.dumps(json_dict).encode("utf-8")) + encryptor.finalize()
        return return_bytes

    def _get_decrypted_bytes(self, response: requests.Response) -> bytes:
        # Get raw data
        encrypted_bytes = response.content
        authorization = response.headers.get("Authorization", None)
        jwt_token = authorization.split()[1]

        # Decrypted aes_key and aes_ctr_nonce
        payload = jwt.decode(jwt=jwt_token, options={"verify_signature": False})
        aes_key = self._master_to_dev_encryption_private_key_obj.decrypt(
            ciphertext=base64.b64decode(payload["aes_key"].encode("ascii")),
            padding=self._get_asymmetric_padding()
        )
        aes_ctr_nonce = self._master_to_dev_encryption_private_key_obj.decrypt(
            ciphertext=base64.b64decode(payload["aes_ctr_nonce"].encode("ascii")),
            padding=self._get_asymmetric_padding()
        )

        # Return decrypted_bytes
        cipher = Cipher(
            algorithm=algorithms.AES(key=aes_key),
            mode=modes.CTR(nonce=aes_ctr_nonce)
        )
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted_bytes) + decryptor.finalize()

    def _get_new_headers(self, aes_key: bytes, aes_ctr_nonce: bytes):
        # Set JWT expiration time to 15 minutes.
        jwt_token = jwt.encode(
            payload={
                "user_id": self._user_id,
                "exp": (datetime.datetime.utcnow() + datetime.timedelta(minutes=15)).timestamp(),
                "aes_key": base64.b64encode(
                    self._dev_to_master_encryption_public_key_obj.encrypt(
                        plaintext=aes_key,
                        padding=self._get_asymmetric_padding()
                    )
                ).decode("ascii"),
                "aes_ctr_nonce": base64.b64encode(
                    self._dev_to_master_encryption_public_key_obj.encrypt(
                        plaintext=aes_ctr_nonce,
                        padding=self._get_asymmetric_padding()
                    )
                ).decode("ascii")
            },
            key=self._dev_to_master_signing_private_key,
            algorithm="RS512"
        )

        return {
            "Authorization": f"Bearer {jwt_token}"
        }

    @staticmethod
    def _get_asymmetric_padding():
        return padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
