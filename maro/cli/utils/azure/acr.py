# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

from maro.cli.utils.subprocess import Subprocess


def login_acr(acr_name: str) -> None:
    command = f"az acr login --name {acr_name}"
    _ = Subprocess.run(command=command)


def list_acr_repositories(acr_name: str) -> list:
    command = f"az acr repository list -n {acr_name}"
    return_str = Subprocess.run(command=command)
    return json.loads(return_str)
