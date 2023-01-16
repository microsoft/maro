# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import pathlib

import yaml


def get_cloud_subscription() -> str:
    return os.environ.get("AZURE_SUBSCRIPTION", "")


def get_user_admin_public_key() -> str:
    with open(os.path.expanduser("~/.ssh/id_rsa.pub"), "r") as fr:
        return fr.read().strip("\n")


if __name__ == "__main__":
    config_details = {
        "cloud/subscription": get_cloud_subscription(),
        "cloud/default_public_key": get_user_admin_public_key(),
    }
    prev_dir_path = pathlib.Path(__file__).parent.parent.absolute()
    with open(os.path.join(prev_dir_path, "config.yml"), "w") as fw:
        yaml.safe_dump(config_details, fw)
