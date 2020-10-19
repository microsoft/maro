# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.grass.utils.copy import copy_files_from_node, copy_files_to_node
from maro.cli.utils.checkers import check_details_validity
from maro.cli.utils.details import load_cluster_details
from maro.cli.utils.lock import lock
from maro.cli.utils.params import GlobalPaths
from maro.utils.exception.cli_exception import CliException


@check_details_validity(mode='grass')
@lock
def push_data(cluster_name: str, local_path: str, remote_path: str, **kwargs):
    # Load details
    cluster_details = load_cluster_details(cluster_name=cluster_name)
    admin_username = cluster_details['user']['admin_username']
    master_public_ip_address = cluster_details['master']['public_ip_address']

    if not remote_path.startswith("/"):
        raise CliException("Invalid remote path")
    copy_files_to_node(
        local_path=local_path,
        remote_dir=f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/data{remote_path}",
        admin_username=admin_username, node_ip_address=master_public_ip_address
    )


@check_details_validity(mode='grass')
@lock
def pull_data(cluster_name: str, local_path: str, remote_path: str, **kwargs):
    # Load details
    cluster_details = load_cluster_details(cluster_name=cluster_name)
    admin_username = cluster_details['user']['admin_username']
    master_public_ip_address = cluster_details['master']['public_ip_address']

    if not remote_path.startswith("/"):
        raise CliException("Invalid remote path")
    copy_files_from_node(
        local_dir=local_path,
        remote_path=f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/data{remote_path}",
        admin_username=admin_username, node_ip_address=master_public_ip_address
    )
