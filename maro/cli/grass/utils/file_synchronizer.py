# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import platform
import shutil
import uuid

from maro.cli.utils.params import GlobalPaths
from maro.cli.utils.path_convertor import PathConvertor
from maro.cli.utils.subprocess import Subprocess
from maro.utils.exception.cli_exception import FileOperationError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class FileSynchronizer:
    """Synchronizer class for files.
    """

    @staticmethod
    def copy_files_to_node(
        local_path: str,
        remote_dir: str,
        node_username: str,
        node_hostname: str,
        node_ssh_port: int
    ) -> None:
        """Copy local files to node, automatically create folder if not exist.

        Args:
            local_path (str): path of the local file.
            remote_dir (str): dir for remote files.
            node_username (str): username of the vm.
            node_hostname (str): hostname of the vm.
            node_ssh_port (int): port of the ssh connection.
        """
        source_path = PathConvertor.build_path_without_trailing_slash(local_path)
        basename = os.path.basename(source_path)
        folder_name = os.path.expanduser(os.path.dirname(source_path))
        target_dir = PathConvertor.build_path_with_trailing_slash(remote_dir)

        mkdir_script = (
            f"ssh -o StrictHostKeyChecking=no -p {node_ssh_port} {node_username}@{node_hostname} "
            f"'mkdir -p {target_dir}'"
        )
        _ = Subprocess.run(command=mkdir_script)

        if platform.system() in ["Linux", "Darwin"]:
            # Copy with pipe
            copy_script = (
                f"tar czf - -C {folder_name} {basename} | "
                f"ssh -o StrictHostKeyChecking=no -p {node_ssh_port} {node_username}@{node_hostname} "
                f"'tar xzf - -C {target_dir}'"
            )
            _ = Subprocess.run(command=copy_script)
        else:
            # Copy with tmp file
            tmp_file_name = uuid.uuid4()
            maro_local_tmp_abs_path = os.path.expanduser(GlobalPaths.MARO_LOCAL_TMP)

            tar_script = f"tar czf {maro_local_tmp_abs_path}/{tmp_file_name} -C {folder_name} {basename}"
            _ = Subprocess.run(command=tar_script)
            copy_script = (
                f"scp {maro_local_tmp_abs_path}/{tmp_file_name} "
                f"{node_username}@{node_hostname}:{GlobalPaths.MARO_LOCAL_TMP}"
            )
            _ = Subprocess.run(command=copy_script)
            untar_script = (
                f"ssh -o StrictHostKeyChecking=no -p {node_ssh_port} {node_username}@{node_hostname} "
                f"'tar xzf {GlobalPaths.MARO_LOCAL_TMP}/{tmp_file_name} -C {target_dir}'"
            )
            _ = Subprocess.run(untar_script)
            remove_script = f"rm {maro_local_tmp_abs_path}/{tmp_file_name}"
            _ = Subprocess.run(remove_script)
            remote_remove_script = (
                f"ssh -o StrictHostKeyChecking=no -p {node_ssh_port} {node_username}@{node_hostname} "
                f"'rm {GlobalPaths.MARO_LOCAL_TMP}/{tmp_file_name}'"
            )
            _ = Subprocess.run(command=remote_remove_script)

    @staticmethod
    def copy_files_from_node(
        local_dir: str,
        remote_path: str,
        node_username: str,
        node_hostname: str,
        node_ssh_port: int
    ) -> None:
        """Copy node files to local, automatically create folder if not exist.

        Args:
            local_dir (str): dir for local files.
            remote_path (str): path of the remote file.
            node_username (str): username of the vm.
            node_hostname (str): hostname of the vm.
            node_ssh_port (int): port of the ssh connection.
        """
        source_path = PathConvertor.build_path_without_trailing_slash(remote_path)
        basename = os.path.basename(source_path)
        folder_name = os.path.dirname(source_path)
        target_dir = PathConvertor.build_path_with_trailing_slash(local_dir)

        # Create local dir
        os.makedirs(os.path.expanduser(target_dir), exist_ok=True)

        if platform.system() in ["Linux", "Darwin"]:
            # Copy with pipe
            copy_script = (
                f"ssh -o StrictHostKeyChecking=no -p {node_ssh_port} {node_username}@{node_hostname} "
                f"'tar czf - -C {folder_name} {basename}' | tar xzf - -C {target_dir}"
            )
            _ = Subprocess.run(command=copy_script)
        else:
            # Copy with tmp file
            tmp_file_name = uuid.uuid4()
            maro_local_tmp_abs_path = os.path.expanduser(GlobalPaths.MARO_LOCAL_TMP)

            tar_script = (
                f"ssh -o StrictHostKeyChecking=no -p {node_ssh_port} {node_username}@{node_hostname} "
                f"tar czf {GlobalPaths.MARO_LOCAL_TMP}/{tmp_file_name} -C {folder_name} {basename}"
            )
            _ = Subprocess.run(command=tar_script)
            copy_script = (
                f"scp {node_username}@{node_hostname}:{GlobalPaths.MARO_LOCAL_TMP}/{tmp_file_name} "
                f"{maro_local_tmp_abs_path}"
            )
            _ = Subprocess.run(command=copy_script)
            untar_script = f"tar xzf {maro_local_tmp_abs_path}/{tmp_file_name} -C {os.path.expanduser(target_dir)}"
            _ = Subprocess.run(command=untar_script)
            remove_script = f"rm {maro_local_tmp_abs_path}/{tmp_file_name}"
            _ = Subprocess.run(command=remove_script)
            remote_remove_script = (
                f"ssh -o StrictHostKeyChecking=no -p {node_ssh_port} {node_username}@{node_hostname} "
                f"'rm {GlobalPaths.MARO_LOCAL_TMP}/{tmp_file_name}'"
            )
            _ = Subprocess.run(command=remote_remove_script)

    @staticmethod
    def copy_and_rename(source_path: str, target_dir: str, new_name: str = None):
        """Copy and rename a file.

        Args:
            source_path (str): path of the source.
            target_dir (str): dir of the target.
            new_name (str): name of the new file, if None, will not do rename.
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
