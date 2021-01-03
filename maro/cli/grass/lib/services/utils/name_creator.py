# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import uuid


class NameCreator:
    @staticmethod
    def create_name_with_uuid(prefix: str, uuid_len: int = 16) -> str:
        postfix = uuid.uuid4().hex[:uuid_len]
        return f"{prefix}{postfix}"

    @staticmethod
    def create_job_id():
        return NameCreator.create_name_with_uuid(prefix="job", uuid_len=8)

    @staticmethod
    def create_component_id():
        return NameCreator.create_name_with_uuid(prefix="component", uuid_len=8)
