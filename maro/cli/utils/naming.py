# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import hashlib
import re
import uuid


def get_valid_file_name(file_name: str):
    return re.sub(r'[\\/*?:"<>|]', "_", file_name)


def generate_name_with_uuid(prefix: str, uuid_len: int = 16) -> str:
    postfix = uuid.uuid4().hex[:uuid_len]
    return f"{prefix}{postfix}"


def generate_name_with_md5(prefix: str, key: str, md5_len: int = 16) -> str:
    postfix = hashlib.md5(key.encode('utf8')).hexdigest()[:md5_len]
    return f"{prefix}{postfix}"


def generate_cluster_id():
    return generate_name_with_uuid(prefix='maro', uuid_len=8)


def generate_node_name():
    return generate_name_with_uuid(prefix='node', uuid_len=8)


def generate_job_id():
    return generate_name_with_uuid(prefix='job', uuid_len=8)


def generate_component_id():
    return generate_name_with_uuid(prefix='component', uuid_len=8)


def generate_image_name():
    return generate_name_with_uuid(prefix='image', uuid_len=8)
