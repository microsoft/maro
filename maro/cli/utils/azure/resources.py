# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

from maro.cli.utils.subprocess import Subprocess


def list_resources(resource_group: str) -> list:
    command = f"az resource list -g {resource_group}"
    return_str = Subprocess.run(command=command)
    return json.loads(return_str)


def delete_resources(resource_ids: list) -> None:
    command = f"az resource delete --ids {' '.join(resource_ids)}"
    _ = Subprocess.run(command=command)


def cleanup(cluster_name: str, resource_group: str) -> None:
    # Get resource list
    resource_list = list_resources(resource_group)

    # Filter resources
    deletable_ids = []
    for resource in resource_list:
        if resource["name"].startswith(cluster_name):
            deletable_ids.append(resource["id"])

    # Delete resources
    if deletable_ids:
        delete_resources(resource_ids=deletable_ids)
