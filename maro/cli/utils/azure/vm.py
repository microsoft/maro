# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

from maro.cli.utils.subprocess import Subprocess


def list_ip_addresses(resource_group: str, vm_name: str) -> list:
    command = f"az vm list-ip-addresses -g {resource_group} --name {vm_name}"
    return_str = Subprocess.run(command=command)
    return json.loads(return_str)


def start_vm(resource_group: str, vm_name: str) -> None:
    command = f"az vm start -g {resource_group} --name {vm_name}"
    _ = Subprocess.run(command=command)


def stop_vm(resource_group: str, vm_name: str) -> None:
    command = f"az vm stop -g {resource_group} --name {vm_name}"
    _ = Subprocess.run(command=command)


def list_vm_sizes(location: str) -> list:
    command = f"az vm list-sizes -l {location}"
    return_str = Subprocess.run(command=command)
    return json.loads(return_str)


def deallocate_vm(resource_group: str, vm_name: str) -> None:
    command = f"az vm deallocate --resource-group {resource_group} --name {vm_name}"
    _ = Subprocess.run(command=command)


def generalize_vm(resource_group: str, vm_name: str) -> None:
    command = f"az vm generalize --resource-group {resource_group} --name {vm_name}"
    _ = Subprocess.run(command=command)


def create_image_from_vm(resource_group: str, image_name: str, vm_name: str) -> None:
    command = f"az image create --resource-group {resource_group} --name {image_name} --source {vm_name}"
    _ = Subprocess.run(command=command)


def get_image_resource_id(resource_group: str, image_name: str) -> str:
    command = f"az image show --resource-group {resource_group} --name {image_name}"
    return_str = Subprocess.run(command=command)
    return json.loads(return_str)["id"]
