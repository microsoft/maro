# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

from maro.cli.utils.subprocess import Subprocess
from maro.utils.exception.cli_exception import CommandExecutionError

from .general import get_resource_client


def get_resource_group(resource_group: str) -> dict:
    command = f"az group show --name {resource_group}"
    try:
        return_str = Subprocess.run(command=command)
        return json.loads(return_str)
    except CommandExecutionError:
        return {}


def delete_resource_group(resource_group: str) -> None:
    command = f"az group delete --yes --name {resource_group}"
    _ = Subprocess.run(command=command)


# Chained Azure resource group operations
def create_resource_group(subscription: str, resource_group: str, location: str):
    """Create the resource group if it does not exist.

    Args:
        subscription (str): Azure subscription name.
        resource group (str): Resource group name.
        location (str): Reousrce group location.

    Returns:
        None.
    """
    resource_client = get_resource_client(subscription)
    return resource_client.resource_groups.create_or_update(resource_group, {"location": location})


def delete_resource_group_under_subscription(subscription: str, resource_group: str):
    resource_client = get_resource_client(subscription)
    return resource_client.resource_groups.begin_delete(resource_group)
