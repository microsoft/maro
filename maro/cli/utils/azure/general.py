# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import subprocess

from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient

from maro.cli.utils.subprocess import Subprocess


def set_subscription(subscription: str) -> None:
    command = f"az account set --subscription {subscription}"
    _ = Subprocess.run(command=command)


def get_version() -> dict:
    command = "az version"
    return_str = Subprocess.run(command=command)
    return json.loads(return_str)


def get_resource_client(subscription: str):
    return ResourceManagementClient(DefaultAzureCredential(), subscription)


def set_env_credentials(dump_path: str, service_principal_name: str):
    os.makedirs(dump_path, exist_ok=True)
    service_principal_file_path = os.path.join(dump_path, f"{service_principal_name}.json")
    # If the service principal file does not exist, create one using the az CLI command.
    # For details on service principals, refer to
    # https://docs.microsoft.com/en-us/azure/active-directory/develop/app-objects-and-service-principals
    if not os.path.exists(service_principal_file_path):
        with open(service_principal_file_path, "w") as fp:
            subprocess.run(
                f"az ad sp create-for-rbac --name {service_principal_name} --sdk-auth --role contributor".split(),
                stdout=fp,
            )

    with open(service_principal_file_path, "r") as fp:
        service_principal = json.load(fp)

    os.environ["AZURE_TENANT_ID"] = service_principal["tenantId"]
    os.environ["AZURE_CLIENT_ID"] = service_principal["clientId"]
    os.environ["AZURE_CLIENT_SECRET"] = service_principal["clientSecret"]
    os.environ["AZURE_SUBSCRIPTION_ID"] = service_principal["subscriptionId"]


def connect_to_aks(resource_group: str, aks: str):
    subprocess.run(f"az aks get-credentials --resource-group {resource_group} --name {aks}".split())


def get_acr_push_permissions(service_principal_id: str, acr: str):
    acr_id = json.loads(
        subprocess.run(f"az acr show --name {acr} --query id".split(), stdout=subprocess.PIPE).stdout,
    )
    subprocess.run(
        f"az role assignment create --assignee {service_principal_id} --scope {acr_id} --role acrpush".split(),
    )
    subprocess.run(f"az acr login --name {acr}".split())
