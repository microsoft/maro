import io
import os
from abc import ABC, abstractmethod
from pathlib import Path

import paramiko
from azure.storage.blob import BlobClient, ContainerClient

from maro.utils.logger import CliLogger

logger = CliLogger(__name__)


class AbsCheckpoint(ABC):
    @abstractmethod
    def set(self, key: str, value: bytes):
        """Set key to hold the value in bytes.

        Args:
            key (str): The key of the k-v pair.
            value (bytes): The value of the k-v pair.

        Returns:
            None.
        """
        logger.info(f"Put key: {key}")

    @abstractmethod
    def get(self, key: str):
        """Get the value with the key.

        Args:
            key (str): The key of the k-v pair.

        Returns:
            bytes: Value in bytes.
        """
        logger.info(f"Get key: {key}")

    @abstractmethod
    def exists(self, key: str):
        """Returns if key exists.

        Args:
            key (str): The key of the k-v pair.

        Returns:
            bool: True if key exists, else false.
        """
        logger.info(f"Get key: {key}")


class AzureBlobCheckpoint(AbsCheckpoint):
    """Checkpoint module base on AzureBlob.

    Make sure you have the permission to access the Blob Service.
    """

    def __init__(self, conn_str: str, container_name: str):
        """Init AzureBlobCheckpoint.

        Args:
            conn_str (str): The connection string for the storage account.
            container_name (str): The name of the container, make sure the container name is valid.
        """
        # Init params
        self._conn_str = conn_str
        self._container_name = container_name

        # Init remote container
        self._create_container_if_not_exist(container=self._container_name)

    def set(self, key: str, value: bytes) -> None:
        super().set(key=key, value=value)
        blob_client = BlobClient.from_connection_string(
            conn_str=self._conn_str,
            container_name=self._container_name,
            blob_name=key
        )
        blob_client.upload_blob(value, overwrite=True)

    def get(self, key: str) -> bytes:
        super().get(key=key)
        blob_client = BlobClient.from_connection_string(
            conn_str=self._conn_str,
            container_name=self._container_name,
            blob_name=key
        )
        blob_data = blob_client.download_blob()
        bytes_io = io.BytesIO()
        blob_data.readinto(bytes_io)
        return bytes_io.getvalue()

    def exists(self, key: str) -> bool:
        super().exists(key=key)
        blob_client = BlobClient.from_connection_string(
            conn_str=self._conn_str,
            container_name=self._container_name,
            blob_name=key
        )
        return blob_client.exists()

    def _create_container_if_not_exist(self, container: str) -> None:
        """Create the container if the target container not exists.

        Args:
            container: target container.

        Returns:
            None.
        """
        container_client = ContainerClient.from_connection_string(
            conn_str=self._conn_str,
            container_name=container
        )
        try:
            container_client.create_container()
        except Exception as e:
            logger.warning_yellow(str(e))


class ServerCheckpoint(AbsCheckpoint):
    """Checkpoint module base on SFTP.

    Make sure you have the permission to access the target folder of the target server.
    """

    def __init__(self, remote_dir: str, admin_username: str, ip_address: str, port: int = 22):
        """Init ServerCheckpoint.

        Args:
            remote_dir (str): Target directory.
            admin_username (str): The username to access the server.
            ip_address (str): The IP address of the server.
            port (int): The access port of the server, default is 22.
        """
        # Init params
        self.remote_dir = remote_dir
        self.admin_username = admin_username
        self.ip_address = ip_address

        # Init SFTPClient instance
        transport = paramiko.Transport((ip_address, port))
        transport.connect(
            username=admin_username,
            pkey=paramiko.RSAKey.from_private_key(open(f"{Path.home()}/.ssh/id_rsa"))
        )
        self._sftp = paramiko.SFTPClient.from_transport(transport)

        # Init remote folder, and cd to remote folder
        self._mkdir_if_not_exist(target_dir=self.remote_dir)
        self._chdir(target_dir=self.remote_dir)

    def set(self, key: str, value: bytes):
        super().set(key=key, value=value)
        bytes_io = io.BytesIO(value)
        self._sftp.putfo(bytes_io, key)

    def get(self, key: str) -> bytes:
        super().get(key=key)
        bytes_io = io.BytesIO()
        self._sftp.getfo(key, bytes_io)
        return bytes_io.getvalue()

    def exists(self, key: str) -> bool:
        super().exists(key=key)
        try:
            self._sftp.open(key)
            return True
        except IOError:
            return False

    def _mkdir_if_not_exist(self, target_dir: str) -> None:
        """Mkdir and chdir if the target directory not exists.

        Args:
            target_dir (str): Target directory.

        Returns:
            None.
        """
        if target_dir == "/":
            self._sftp.chdir("/")
            return
        if target_dir == "":
            return
        try:
            self._sftp.chdir(target_dir)
        except IOError:
            logger.debug(target_dir)
            dir_name, base_name = os.path.split(target_dir.rstrip("/"))
            self._mkdir_if_not_exist(dir_name)
            self._sftp.mkdir(base_name)
            self._sftp.chdir(base_name)
            return

    def _chdir(self, target_dir: str) -> None:
        """Change directory to the target path.

        Args:
            target_dir (str): Target directory.

        Returns:
            None
        """
        self._sftp.chdir(target_dir)
