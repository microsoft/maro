# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import secrets
import shutil
import string

import yaml

from maro.cli.grass.executors.grass_executor import GrassExecutor
from maro.cli.grass.utils.file_synchronizer import FileSynchronizer
from maro.cli.grass.utils.master_api_client import MasterApiClientV1
from maro.cli.grass.utils.params import GrassParams, GrassPaths
from maro.cli.utils.deployment_validator import DeploymentValidator
from maro.cli.utils.details_reader import DetailsReader
from maro.cli.utils.name_creator import NameCreator
from maro.cli.utils.params import GlobalParams, GlobalPaths
from maro.utils.exception.cli_exception import BadRequestError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class GrassOnPremisesExecutor(GrassExecutor):
    """Executor for grass/on-premises mode.

    See https://maro.readthedocs.io/en/latest/key_components/orchestration.html for reference.
    """

    def __init__(self, cluster_name: str):
        super().__init__(cluster_details=DetailsReader.load_cluster_details(cluster_name=cluster_name))

    @staticmethod
    def create(create_deployment: dict):
        """Create MARO Cluster with create_deployment.

        Args:
            create_deployment (dict): create_deployment of grass/on-premises.
                See lib/deployments/internal for reference.

        Returns:
            None.
        """
        logger.info("Creating cluster")

        # Get standardized cluster_details
        cluster_details = GrassOnPremisesExecutor._standardize_cluster_details(create_deployment=create_deployment)
        cluster_name = cluster_details["name"]
        if os.path.isdir(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}"):
            raise BadRequestError(f"Cluster '{cluster_name}' is exist")

        # Start creating
        try:
            GrassOnPremisesExecutor._init_master(cluster_details=cluster_details)
            GrassOnPremisesExecutor._create_user(cluster_details=cluster_details)

            # Remote create master, cluster after initialization
            master_api_client = MasterApiClientV1(
                master_hostname=cluster_details["master"]["public_ip_address"],
                master_api_server_port=cluster_details["master"]["api_server"]["port"],
                user_id=cluster_details["user"]["id"],
                master_to_dev_encryption_private_key=cluster_details["user"]["master_to_dev_encryption_private_key"],
                dev_to_master_encryption_public_key=cluster_details["user"]["dev_to_master_encryption_public_key"],
                dev_to_master_signing_private_key=cluster_details["user"]["dev_to_master_signing_private_key"]
            )
            master_api_client.create_master(master_details=cluster_details["master"])
            master_api_client.create_cluster(cluster_details=cluster_details)
        except Exception as e:
            # If failed, remove details folder, then raise
            shutil.rmtree(path=f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}")
            logger.error_red(f"Failed to create cluster '{cluster_name}'")
            raise e

        logger.info_green(f"Cluster {cluster_name} has been created.")

    @staticmethod
    def _standardize_cluster_details(create_deployment: dict) -> dict:
        """Standardize cluster_details from create_deployment.

        We use create_deployment to build cluster_details (they share the same keys structure).

        Args:
            create_deployment (dict): create_deployment of grass/on-premises.
                See lib/deployments/internal for reference.

        Returns:
            dict: standardized cluster_details.
        """
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

    # maro grass delete

    def delete(self):
        """Delete the MARO Cluster.

        Leave all nodes in the MARO Cluster, then delete MARO Master.

        Returns:
            None.
        """
        logger.info(f"Deleting cluster '{self.cluster_name}'")

        nodes_details = self.master_api_client.list_nodes()
        for node_details in nodes_details:
            self.remote_leave_cluster(
                node_username=node_details["username"],
                node_hostname=node_details["public_ip_address"],
                node_ssh_port=node_details["ssh"]["port"]
            )

        self.remote_delete_master(
            master_username=self.master_username,
            master_hostname=self.master_public_ip_address,
            master_ssh_port=self.master_ssh_port
        )

        shutil.rmtree(path=f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}")

        logger.info_green(f"Cluster '{self.cluster_name}' is deleted")

    # maro grass join

    @staticmethod
    def join_cluster(join_cluster_deployment: dict):
        """Entry method for join_cluster.

        Args:
            join_cluster_deployment (dict): join_cluster_deployment of grass/on-premises.
                See lib/deployments/internal for reference.

        Returns:
            None.
        """
        GrassOnPremisesExecutor._join_cluster(join_cluster_deployment=join_cluster_deployment)

    @staticmethod
    def _join_cluster(join_cluster_deployment: dict):
        """Join a vm to the MARO Cluster with join_cluster_deployment.

        Args:
            join_cluster_deployment (dict): join_cluster_deployment of grass/on-premises.
                See lib/deployments/internal for reference.

        Returns:
            None.
        """
        logger.info("Joining the cluster")

        # Get standardized join_cluster_deployment
        join_cluster_deployment = GrassOnPremisesExecutor._standardize_join_cluster_deployment(
            join_cluster_deployment=join_cluster_deployment
        )

        # Save join_cluster_deployment TODO: do checking, already join another node
        with open(file=f"{GlobalPaths.ABS_MARO_LOCAL_TMP}/join_cluster_deployment.yml", mode="w") as fw:
            yaml.safe_dump(data=join_cluster_deployment, stream=fw)

        # Copy required files
        local_path_to_remote_dir = {
            f"{GlobalPaths.ABS_MARO_LOCAL_TMP}/join_cluster_deployment.yml": GlobalPaths.MARO_LOCAL_TMP
        }
        for local_path, remote_dir in local_path_to_remote_dir.items():
            FileSynchronizer.copy_files_to_node(
                local_path=local_path,
                remote_dir=remote_dir,
                node_username=join_cluster_deployment["node"]["username"],
                node_hostname=join_cluster_deployment["node"]["public_ip_address"],
                node_ssh_port=join_cluster_deployment["node"]["ssh"]["port"]
            )

        # Remote join node
        GrassOnPremisesExecutor.remote_join_cluster(
            node_username=join_cluster_deployment["node"]["username"],
            node_hostname=join_cluster_deployment["node"]["public_ip_address"],
            node_ssh_port=join_cluster_deployment["node"]["ssh"]["port"],
            master_private_ip_address=join_cluster_deployment["master"]["private_ip_address"],
            master_api_server_port=join_cluster_deployment["master"]["api_server"]["port"],
            deployment_path=f"{GlobalPaths.MARO_LOCAL_TMP}/join_cluster_deployment.yml"
        )

        os.remove(f"{GlobalPaths.ABS_MARO_LOCAL_TMP}/join_cluster_deployment.yml")

        logger.info_green("Node is joined to the cluster")

    @staticmethod
    def _standardize_join_cluster_deployment(join_cluster_deployment: dict) -> dict:
        """Standardize join_cluster_deployment.

        Args:
            join_cluster_deployment (dict): join_cluster_deployment of grass/on-premises.
                See lib/deployments/internal for reference.

        Returns:
            dict: standardized join_cluster_deployment.
        """
        optional_key_to_value = {
            "root['master']['redis']": {"port": GlobalParams.DEFAULT_REDIS_PORT},
            "root['master']['redis']['port']": GlobalParams.DEFAULT_REDIS_PORT,
            "root['master']['api_server']": {"port": GrassParams.DEFAULT_API_SERVER_PORT},
            "root['master']['api_server']['port']": GrassParams.DEFAULT_API_SERVER_PORT,
            "root['node']['resources']": {
                "cpu": "all",
                "memory": "all",
                "gpu": "all"
            },
            "root['node']['resources']['cpu']": "all",
            "root['node']['resources']['memory']": "all",
            "root['node']['resources']['gpu']": "all",
            "root['node']['api_server']": {"port": GrassParams.DEFAULT_API_SERVER_PORT},
            "root['node']['api_server']['port']": GrassParams.DEFAULT_API_SERVER_PORT,
            "root['node']['ssh']": {"port": GlobalParams.DEFAULT_SSH_PORT},
            "root['node']['ssh']['port']": GlobalParams.DEFAULT_SSH_PORT,
            "root['configs']": {
                "install_node_runtime": False,
                "install_node_gpu_support": False
            },
            "root['configs']['install_node_runtime']": False,
            "root['configs']['install_node_gpu_support']": False
        }
        with open(
            file=f"{GrassPaths.ABS_MARO_GRASS_LIB}/deployments/internal/grass_on_premises_join_cluster.yml",
            mode="r"
        ) as fr:
            create_deployment_template = yaml.safe_load(stream=fr)

        DeploymentValidator.validate_and_fill_dict(
            template_dict=create_deployment_template,
            actual_dict=join_cluster_deployment,
            optional_key_to_value=optional_key_to_value
        )

        return join_cluster_deployment

    # maro grass leave

    @staticmethod
    def leave(leave_cluster_deployment: dict) -> None:
        """Join a vm from the MARO Cluster with leave_cluster_deployment.

        Args:
            leave_cluster_deployment (dict): leave_cluster_deployment of grass/on-premises.
                See lib/deployments/internal for reference.

        Returns:
            None.
        """
        logger.info("Node is leaving")

        if not leave_cluster_deployment:
            # Local leave node
            GrassOnPremisesExecutor.local_leave_cluster()
        else:
            # Remote leave node
            GrassOnPremisesExecutor.remote_leave_cluster(
                node_username=leave_cluster_deployment["node"]["username"],
                node_hostname=leave_cluster_deployment["node"]["hostname"],
                node_ssh_port=leave_cluster_deployment["node"]["ssh"]["port"]
            )

        logger.info_green("Node is left")
