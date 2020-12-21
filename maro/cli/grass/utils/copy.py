# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import platform
import shutil
import uuid

from maro.cli.utils.copy import get_reformatted_source_path, get_reformatted_target_dir
from maro.cli.utils.params import GlobalPaths
from maro.cli.utils.subprocess import SubProcess
from maro.utils.exception.cli_exception import FileOperationError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


def copy_files_to_node(local_path: str, remote_dir: str, admin_username: str, node_ip_address: str) -> None:
    """Copy local files to node, automatically create folder if not exist.

    Args:
        local_path (str): path of the local file
        remote_dir (str): dir for remote files
        admin_username (str)
        node_ip_address (str)
    """
    source_path = get_reformatted_source_path(local_path)
    basename = os.path.basename(source_path)
    folder_name = os.path.expanduser(os.path.dirname(source_path))
    target_dir = get_reformatted_target_dir(remote_dir)

    mkdir_script = f"ssh -o StrictHostKeyChecking=no {admin_username}@{node_ip_address} 'mkdir -p {target_dir}'"
    _ = SubProcess.run(mkdir_script)

    if platform.system() == "Linux":
        # Copy with pipe
        copy_script = (
            f"tar czf - -C {folder_name} {basename} | "
            f"ssh -o StrictHostKeyChecking=no {admin_username}@{node_ip_address} 'tar xzf - -C {target_dir}'"
        )
        _ = SubProcess.run(copy_script)
    else:
        # Copy with tmp file
        tmp_file_name = uuid.uuid4()
        maro_local_tmp_abs_path = os.path.expanduser(GlobalPaths.MARO_LOCAL_TMP)

        tar_script = f"tar czf {maro_local_tmp_abs_path}/{tmp_file_name} -C {folder_name} {basename}"
        _ = SubProcess.run(tar_script)
        copy_script = (
            f"scp {maro_local_tmp_abs_path}/{tmp_file_name} "
            f"{admin_username}@{node_ip_address}:{GlobalPaths.MARO_LOCAL_TMP}"
        )
        _ = SubProcess.run(copy_script)
        untar_script = (
            f"ssh -o StrictHostKeyChecking=no {admin_username}@{node_ip_address} "
            f"'tar xzf {GlobalPaths.MARO_LOCAL_TMP}/{tmp_file_name} -C {target_dir}'"
        )
        _ = SubProcess.run(untar_script)
        remove_script = f"rm {maro_local_tmp_abs_path}/{tmp_file_name}"
        _ = SubProcess.run(remove_script)
        remote_remove_script = (
            f"ssh -o StrictHostKeyChecking=no {admin_username}@{node_ip_address} "
            f"'rm {GlobalPaths.MARO_LOCAL_TMP}/{tmp_file_name}'"
        )
        _ = SubProcess.run(remote_remove_script)


def copy_files_from_node(local_dir: str, remote_path: str, admin_username: str, node_ip_address: str) -> None:
    """Copy node files to local, automatically create folder if not exist.

    Args:
        local_dir (str): dir for local files
        remote_path (str): path of the remote file
        admin_username (str)
        node_ip_address (str)
    """
    source_path = get_reformatted_source_path(remote_path)
    basename = os.path.basename(source_path)
    folder_name = os.path.dirname(source_path)
    target_dir = get_reformatted_target_dir(local_dir)

    # Create local dir
    os.makedirs(os.path.expanduser(target_dir), exist_ok=True)

    if platform.system() == "Linux":
        # Copy with pipe
        copy_script = (
            f"ssh -o StrictHostKeyChecking=no {admin_username}@{node_ip_address} "
            f"'tar czf - -C {folder_name} {basename}' | tar xzf - -C {target_dir}"
        )
        _ = SubProcess.run(copy_script)
    else:
        # Copy with tmp file
        tmp_file_name = uuid.uuid4()
        maro_local_tmp_abs_path = os.path.expanduser(GlobalPaths.MARO_LOCAL_TMP)

        tar_script = (
            f"ssh -o StrictHostKeyChecking=no {admin_username}@{node_ip_address} "
            f"tar czf {GlobalPaths.MARO_LOCAL_TMP}/{tmp_file_name} -C {folder_name} {basename}"
        )
        _ = SubProcess.run(tar_script)
        copy_script = (
            f"scp {admin_username}@{node_ip_address}:{GlobalPaths.MARO_LOCAL_TMP}/{tmp_file_name} "
            f"{maro_local_tmp_abs_path}"
        )
        _ = SubProcess.run(copy_script)
        untar_script = f"tar xzf {maro_local_tmp_abs_path}/{tmp_file_name} -C {os.path.expanduser(target_dir)}"
        _ = SubProcess.run(untar_script)
        remove_script = f"rm {maro_local_tmp_abs_path}/{tmp_file_name}"
        _ = SubProcess.run(remove_script)
        remote_remove_script = (
            f"ssh -o StrictHostKeyChecking=no {admin_username}@{node_ip_address} "
            f"'rm {GlobalPaths.MARO_LOCAL_TMP}/{tmp_file_name}'"
        )
        _ = SubProcess.run(remote_remove_script)


def copy_and_rename(source_path: str, target_dir: str, new_name: str = None):
    """Copy and rename a file.

    Args:
        source_path (str): path of the source
        target_dir (str): dir of the target
        new_name (str): name of the new file, if None, will not do rename
    """
    source_path = os.path.expanduser(source_path)
    target_dir = os.path.expanduser(target_dir)

    if os.path.isdir(source_path):
        raise FileOperationError(f"Cannot be a folder: '{source_path}'.")
    shutil.copy2(source_path, target_dir)

    if new_name is not None:
        old_name = os.path.basename(source_path)
        old_target_path = os.path.join(target_dir, old_name)
        new_target_path = os.path.join(target_dir, new_name)
        os.rename(old_target_path, new_target_path)
