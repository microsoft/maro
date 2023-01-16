# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import datetime
import json
import os
from typing import Union

from azure.core.exceptions import ResourceExistsError
from azure.storage.fileshare import ShareClient, ShareDirectoryClient

from maro.cli.utils.subprocess import Subprocess


def get_storage_account_keys(resource_group: str, storage_account_name: str) -> dict:
    command = f"az storage account keys list -g {resource_group} --account-name {storage_account_name}"
    return_str = Subprocess.run(command=command)
    return json.loads(return_str)


def get_storage_account_sas(
    account_name: str,
    services: str = "bqtf",
    resource_types: str = "sco",
    permissions: str = "rwdlacup",
    expiry: str = (datetime.datetime.utcnow() + datetime.timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
) -> str:
    command = (
        f"az storage account generate-sas --account-name {account_name} --services {services} "
        f"--resource-types {resource_types} --permissions {permissions} --expiry {expiry}"
    )
    sas_str = Subprocess.run(command=command).strip("\n").replace('"', "")
    # logger.debug(sas_str)
    return sas_str


def get_connection_string(storage_account_name: str) -> str:
    """Get the connection string for a storage account.

    Args:
        storage_account_name: The storage account name.

    Returns:
        str: Connection string.
    """
    command = f"az storage account show-connection-string --name {storage_account_name}"
    return_str = Subprocess.run(command=command)
    return json.loads(return_str)["connectionString"]


def get_fileshare(storage_account_name: str, fileshare_name: str):
    connection_string = get_connection_string(storage_account_name)
    share = ShareClient.from_connection_string(connection_string, fileshare_name)
    try:
        share.create_share()
    except ResourceExistsError:
        pass

    return share


def get_directory(share: Union[ShareClient, ShareDirectoryClient], name: str):
    if isinstance(share, ShareClient):
        directory = share.get_directory_client(directory_path=name)
        try:
            directory.create_directory()
        except ResourceExistsError:
            pass

        return directory
    elif isinstance(share, ShareDirectoryClient):
        try:
            return share.create_subdirectory(name)
        except ResourceExistsError:
            return share.get_subdirectory_client(name)


def upload_to_fileshare(share: Union[ShareClient, ShareDirectoryClient], source_path: str, name: str = None):
    if os.path.isdir(source_path):
        if not name:
            name = os.path.basename(source_path)
        directory = get_directory(share, name)
        for file in os.listdir(source_path):
            upload_to_fileshare(directory, os.path.join(source_path, file))
    else:
        with open(source_path, "rb") as fp:
            share.upload_file(file_name=os.path.basename(source_path), data=fp)


def download_from_fileshare(share: ShareDirectoryClient, file_name: str, local_path: str):
    file = share.get_file_client(file_name=file_name)
    with open(local_path, "wb") as fp:
        fp.write(file.download_file().readall())


def delete_directory(share: Union[ShareClient, ShareDirectoryClient], name: str, recursive: bool = True):
    share.delete_directory(directory_name=name)
