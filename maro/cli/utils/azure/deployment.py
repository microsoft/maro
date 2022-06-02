# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .general import get_resource_client


def create_deployment(
    subscription: str,
    resource_group: str,
    deployment_name: str,
    template: dict,
    params: dict,
    sync: bool = True,
) -> None:
    params = {k: {"value": v} for k, v in params.items()}
    resource_client = get_resource_client(subscription)
    deployment_params = {"mode": "Incremental", "template": template, "parameters": params}
    result = resource_client.deployments.begin_create_or_update(
        resource_group,
        deployment_name,
        {"properties": deployment_params},
    )
    if sync:
        result.result()


def delete_deployment(subscription: str, resource_group: str, deployment_name: str) -> None:
    resource_client = get_resource_client(subscription)
    resource_client.deployments.begin_delete(resource_group, deployment_name)
