# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import shutil

from maro.cli.utils.subprocess import SubProcess
from maro.utils.exception.cli_exception import CliException
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


def copy_files_to_node(local_path: str, remote_dir: str, admin_username: str, node_ip_address: str):
    """Copy local files to node, automatically create folder if not exist

    Args:
        local_path (str): path of the local file
        remote_dir (str): dir for remote files
        admin_username (str)
        node_ip_address (str)
    """
    copy_scripts = f"rsync -e 'ssh -o StrictHostKeyChecking=no' " \
                   f"-az {local_path} {admin_username}@{node_ip_address}:{remote_dir}"
    _ = SubProcess.run(copy_scripts)


def copy_files_from_node(local_dir: str, remote_path: str, admin_username: str, node_ip_address: str):
    """Copy node files to local, automatically create folder if not exist

    Args:
        local_dir (str): dir for local files
        remote_path (str): path of the remote file
        admin_username (str)
        node_ip_address (str)
    """
    copy_scripts = f"rsync -e 'ssh -o StrictHostKeyChecking=no' " \
                   f"-az {admin_username}@{node_ip_address}:{remote_path} {local_dir}"
    _ = SubProcess.run(copy_scripts)


def sync_mkdir(remote_path: str, admin_username: str, node_ip_address: str):
    """mkdir synchronously at local and remote

    Args:
        remote_path (str): path of the remote file
        admin_username (str)
        node_ip_address (str)
    """
    command = f"mkdir -p {remote_path}"
    _ = SubProcess.run(command)

    command = f"ssh -o StrictHostKeyChecking=no {admin_username}@{node_ip_address} 'mkdir -p {remote_path}'"
    _ = SubProcess.run(command)


def copy_and_rename(source_path: str, target_dir: str, new_name: str = None):
    """Copy and rename a file

    Args:
        source_path (str): path of the source
        target_dir (str): dir of the target
        new_name (str): name of the new file, if None, will not do rename

    Raises:
        CliException:
    """
    source_path = os.path.expanduser(source_path)
    target_dir = os.path.expanduser(target_dir)

    if os.path.isdir(source_path):
        raise CliException("Invalid file path: cannot be a folder")
    shutil.copy2(source_path, target_dir)

    if new_name is not None:
        old_name = os.path.basename(source_path)
        old_target_path = os.path.join(target_dir, old_name)
        new_target_path = os.path.join(target_dir, new_name)
        os.rename(old_target_path, new_target_path)
