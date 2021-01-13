# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import secrets
import shutil
import string
import time

import yaml

from maro.cli.grass.executors.grass_executor import GrassExecutor
from maro.cli.grass.utils.file_synchronizer import FileSynchronizer
from maro.cli.grass.utils.master_api_client import MasterApiClientV1
from maro.cli.grass.utils.params import GrassParams, GrassPaths
from maro.cli.utils.deployment_validator import DeploymentValidator
from maro.cli.utils.details_reader import DetailsReader
from maro.cli.utils.details_writer import DetailsWriter
from maro.cli.utils.name_creator import NameCreator
from maro.cli.utils.params import GlobalParams, GlobalPaths
from maro.utils.exception.cli_exception import BadRequestError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class GrassOnPremisesExecutor(GrassExecutor):

    def __init__(self, cluster_name: str):
        super().__init__(cluster_details=DetailsReader.load_cluster_details(cluster_name=cluster_name))

    @staticmethod
    def create(create_deployment: dict):
        logger.info("Creating cluster")

        # Get standardized cluster_details
        cluster_details = GrassOnPremisesExecutor._standardize_cluster_details(create_deployment=create_deployment)
        cluster_name = cluster_details["name"]
        if os.path.isdir(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}"):
            raise BadRequestError(f"Cluster '{cluster_name}' is exist")

        # Start creating
        try:
            GrassOnPremisesExecutor._init_master(cluster_details=cluster_details)
        except Exception as e:
            # If failed, remove details folder, then raise
            shutil.rmtree(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}"))
            logger.error_red(f"Failed to create cluster '{cluster_name}'")
            raise e

        logger.info_green(f"Cluster {cluster_name} has been created.")

    @staticmethod
    def _standardize_cluster_details(create_deployment: dict) -> dict:
        samba_password = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20))
        optional_key_to_value = {
            "root['master']['redis']": {"port": GlobalParams.DEFAULT_REDIS_PORT},
            "root['master']['redis']['port']": GlobalParams.DEFAULT_REDIS_PORT,
            "root['master']['fluentd']": {"port": GlobalParams.DEFAULT_FLUENTD_PORT},
            "root['master']['fluentd']['port']": GlobalParams.DEFAULT_FLUENTD_PORT,
            "root['master']['samba']": {
                "password": samba_password
            },
            "root['master']['samba']['password']": samba_password,
            "root['master']['ssh']": {"port": GlobalParams.DEFAULT_SSH_PORT},
            "root['master']['ssh']['port']": GlobalParams.DEFAULT_SSH_PORT,
            "root['master']['api_server']": {"port": GrassParams.DEFAULT_API_SERVER_PORT},
            "root['master']['api_server']['port']": GrassParams.DEFAULT_API_SERVER_PORT
        }
        with open(f"{GrassPaths.ABS_MARO_GRASS_LIB}/deployments/internal/grass_on_premises_create.yml") as fr:
            create_deployment_template = yaml.safe_load(fr)
        DeploymentValidator.validate_and_fill_dict(
            template_dict=create_deployment_template,
            actual_dict=create_deployment,
            optional_key_to_value=optional_key_to_value
        )

        # Init runtime fields.
        create_deployment["id"] = NameCreator.create_cluster_id()
        create_deployment["master"]["image_files"] = {}

        return create_deployment

    @staticmethod
    def _init_master(cluster_details: dict):
        logger.info("Initializing Master VM")

        # Make sure master is able to connect
        GrassOnPremisesExecutor.retry_connection(
            node_username=cluster_details["master"]["username"],
            node_hostname=cluster_details["master"]["public_ip_address"],
            node_ssh_port=cluster_details["master"]["ssh"]["port"]
        )

        DetailsWriter.save_cluster_details(
            cluster_name=cluster_details["name"],
            cluster_details=cluster_details
        )

        # Copy required files
        local_path_to_remote_dir = {
            GrassPaths.MARO_GRASS_LIB: f"{GlobalPaths.MARO_SHARED}/lib",
            f"{GlobalPaths.MARO_CLUSTERS}/{cluster_details['name']}": f"{GlobalPaths.MARO_SHARED}/clusters"
        }
        for local_path, remote_dir in local_path_to_remote_dir.items():
            FileSynchronizer.copy_files_to_node(
                local_path=local_path,
                remote_dir=remote_dir,
                node_username=cluster_details["master"]["username"],
                node_hostname=cluster_details["master"]["public_ip_address"],
                node_ssh_port=cluster_details["master"]["ssh"]["port"]
            )

        # Remote init master
        GrassOnPremisesExecutor.remote_init_master(
            master_username=cluster_details["master"]["username"],
            master_hostname=cluster_details["master"]["public_ip_address"],
            master_ssh_port=cluster_details["master"]["ssh"]["port"],
            cluster_name=cluster_details["name"]
        )
        # Gracefully wait
        time.sleep(10)

        # Init master_api_client and remote create master
        master_api_client = MasterApiClientV1(
            master_hostname=cluster_details["master"]["public_ip_address"],
            master_api_server_port=cluster_details["master"]["api_server"]["port"]
        )
        master_api_client.create_master(master_details=cluster_details["master"])

        logger.info_green("Master VM is initialized")

    # maro grass delete

    def delete(self):
        logger.info(f"Deleting cluster '{self.cluster_name}'")

        nodes_details = self.master_api_client.list_nodes()
        for node_details in nodes_details:
            self.remote_leave_node(
                node_username=node_details["username"],
                node_hostname=node_details["public_ip_address"],
                node_ssh_port=node_details["ssh"]["port"]
            )

        self.remote_release_master(
            master_username=self.master_username,
            master_hostname=self.master_public_ip_address,
            master_ssh_port=self.master_ssh_port
        )

        shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}")

        logger.info_green(f"Cluster '{self.cluster_name}' is deleted")

    # maro grass join

    @staticmethod
    def join_node(join_node_deployment: dict):
        GrassOnPremisesExecutor._join_node(join_node_deployment=join_node_deployment)

    @staticmethod
    def _join_node(join_node_deployment: dict):
        logger.info(f"Joining node")

        # Save join_node_deployment TODO: do checking, already join another node
        with open(f"{GlobalPaths.ABS_MARO_LOCAL_TMP}/join_node.yml", "w") as fw:
            yaml.safe_dump(data=join_node_deployment, stream=fw)

        # Copy required files
        local_path_to_remote_dir = {f"{GlobalPaths.MARO_LOCAL_TMP}/join_node.yml": "~/"}
        for local_path, remote_dir in local_path_to_remote_dir.items():
            FileSynchronizer.copy_files_to_node(
                local_path=local_path,
                remote_dir=remote_dir,
                node_username=join_node_deployment["node"]["username"],
                node_hostname=join_node_deployment["node"]["public_ip_address"],
                node_ssh_port=join_node_deployment["node"]["ssh"]["port"]
            )

        # Remote join node
        GrassOnPremisesExecutor.remote_join_node(
            node_username=join_node_deployment["node"]["username"],
            node_hostname=join_node_deployment["node"]["public_ip_address"],
            node_ssh_port=join_node_deployment["node"]["ssh"]["port"],
            master_hostname=join_node_deployment["master"]["hostname"],
            master_api_server_port=join_node_deployment["master"]["api_server"]["port"],
            deployment_path=f"~/join_node.yml"
        )

        os.remove(f"{GlobalPaths.ABS_MARO_LOCAL_TMP}/join_node.yml")

        logger.info_green(f"Node is joined")

    # maro grass leave

    @staticmethod
    def leave(leave_node_deployment: dict) -> None:
        logger.info(f"Node is leaving")

        if not leave_node_deployment:
            # Local leave node
            GrassOnPremisesExecutor.local_leave_node()
        else:
            # Remote leave node
            GrassOnPremisesExecutor.remote_leave_node(
                node_username=leave_node_deployment["node"]["username"],
                node_hostname=leave_node_deployment["node"]["hostname"],
                node_ssh_port=leave_node_deployment["node"]["ssh"]["port"]
            )

        logger.info_green(f"Node is left")
