import io
import os
from abc import ABC, abstractmethod
from pathlib import Path

import paramiko
from azure.storage.blob import ContainerClient, BlobClient

from maro.utils.logger import CliLogger

logger = CliLogger(__name__)


class AbsCheckpoint(ABC):
    @abstractmethod
    def set_data(self, key: str, value: bytes):
        """Set key to hold the value in bytes.

        Args:
            key (str): The key of the k-v pair.
            value (bytes): The value of the k-v pair.

        Returns:
            None.
        """
        logger.info(f"Put key: {key}")

    @abstractmethod
    def get_data(self, key: str):
        """Get the value with the key.

        Args:
            key (str): The key of the k-v pair.

        Returns:
            bytes: Value in bytes.
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

    def set_data(self, key: str, value: bytes) -> None:
        super().set_data(key, value)
        blob_client = BlobClient.from_connection_string(
            conn_str=self._conn_str,
            container_name=self._container_name,
            blob_name=key
        )
        blob_client.upload_blob(value, overwrite=True)

    def get_data(self, key: str) -> bytes:
        super().get_data(key)
        blob_client = BlobClient.from_connection_string(
            conn_str=self._conn_str,
            container_name=self._container_name,
            blob_name=key
        )
        blob_data = blob_client.download_blob()
        bytes_io = io.BytesIO()
        blob_data.readinto(bytes_io)
        return bytes_io.getvalue()

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

        # Init remote folder
        self._mkdir_if_not_exist(target_dir=self.remote_dir)

    def set_data(self, key: str, value: bytes):
        super().set_data(key, value)
        bytes_io = io.BytesIO(value)
        self._sftp.putfo(bytes_io, key)

    def get_data(self, key: str) -> bytes:
        super().get_data(key)
        bytes_io = io.BytesIO()
        self._sftp.getfo(key, bytes_io)
        return bytes_io.getvalue()

    def _mkdir_if_not_exist(self, target_dir: str) -> None:
        """Mkdir if the target directory not exists.

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
