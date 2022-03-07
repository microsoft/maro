# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

FILE_SUFFIX = "ckpt"


def extract_algo_inst_name(policy_name: str) -> str:
    """Extract the algorithm instance's name from the policy name.

    Args:
        policy_name (str): Policy name.

    Returns:
        algo_inst_name (str)
    """
    return policy_name.split(".")[0]


def get_training_state_path(dir_path: str, algo_inst_name: str) -> str:
    return os.path.join(dir_path, f"{algo_inst_name}.{FILE_SUFFIX}")
