# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import hashlib


def get_checksum(file_path: str, block_size=128):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size * md5.block_size), b''):
            md5.update(chunk)
    return md5.hexdigest()
