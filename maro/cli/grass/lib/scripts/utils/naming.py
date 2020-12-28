# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import uuid


def generate_name_with_uuid(prefix: str, uuid_len: int = 16) -> str:
    postfix = uuid.uuid4().hex[:uuid_len]
    return f"{prefix}{postfix}"
