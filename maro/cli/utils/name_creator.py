# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import hashlib
import re
import uuid


class NameCreator:
    @staticmethod
    def get_valid_file_name(file_name: str):
        return re.sub(r"[\\/*?:\"<>|]", "_", file_name)

    @staticmethod
    def create_name_with_uuid(prefix: str, uuid_len: int = 16) -> str:
        postfix = uuid.uuid4().hex[:uuid_len]
        return f"{prefix}{postfix}"

    @staticmethod
    def create_name_with_md5(prefix: str, key: str, md5_len: int = 16) -> str:
        postfix = hashlib.md5(key.encode("utf8")).hexdigest()[:md5_len]
        return f"{prefix}{postfix}"

    @staticmethod
    def create_cluster_id():
        return NameCreator.create_name_with_uuid(prefix="maro", uuid_len=8)

    @staticmethod
    def create_node_name():
        return NameCreator.create_name_with_uuid(prefix="node", uuid_len=8)

    @staticmethod
    def create_job_id():
        return NameCreator.create_name_with_uuid(prefix="job", uuid_len=8)

    @staticmethod
    def create_component_id():
        return NameCreator.create_name_with_uuid(prefix="component", uuid_len=8)

    @staticmethod
    def create_schedule_id():
        return NameCreator.create_name_with_uuid(prefix="schedule", uuid_len=8)
