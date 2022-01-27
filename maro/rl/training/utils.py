# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

FILE_SUFFIX = "ckpt"


def extract_trainer_name(policy_name: str) -> str:
    return policy_name.split(".")[0]


def get_trainer_state_path(dir_path: str, trainer_name: str) -> str:
    return os.path.join(dir_path, f"{trainer_name}.{FILE_SUFFIX}")
