# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

import yaml

from maro.cli.grass.utils.copy import copy_files_to_node
from maro.cli.utils.params import GlobalPaths
from maro.utils.logger import CliLogger

logger = CliLogger(__name__)


def save_cluster_details(cluster_name: str, cluster_details: dict, sync: bool = True) -> None:
    with open(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/details.yml"), 'w') as fw:
        yaml.safe_dump(cluster_details, fw)
    if sync:
        _sync_cluster_details(cluster_details=cluster_details)


def load_cluster_details(cluster_name: str, sync: bool = False) -> dict:
    with open(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/details.yml"), 'r') as fr:
        cluster_details = yaml.safe_load(fr)
    if sync:
        _sync_cluster_details(cluster_details=cluster_details)
    return cluster_details


def save_job_details(cluster_name: str, job_name: str, job_details: dict, sync: bool = True) -> None:
    with open(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/jobs/{job_name}/details.yml"), 'w') as fw:
        yaml.safe_dump(job_details, fw)
    if sync:
        _sync_job_details(cluster_name=cluster_name,
                          job_name=job_name, job_details=job_details)


def load_job_details(cluster_name: str, job_name: str, sync: bool = False) -> dict:
    with open(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/jobs/{job_name}/details.yml"), 'r') as fr:
        details = yaml.safe_load(fr)
    if sync:
        _sync_job_details(cluster_name=cluster_name,
                          job_name=job_name, job_details=details)
    return details


def save_schedule_details(cluster_name: str, schedule_name: str, schedule_details: dict, sync: bool = True) -> None:
    with open(
        os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/schedules/{schedule_name}/details.yml"),
        'w'
    ) as fw:
        yaml.safe_dump(schedule_details, fw)


def load_schedule_details(cluster_name: str, schedule_name: str, sync: bool = False) -> dict:
    with open(
        os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/schedules/{schedule_name}/details.yml"),
        'r'
    ) as fr:
        details = yaml.safe_load(fr)
    return details


def _sync_cluster_details(cluster_details: dict) -> None:
    try:
        cluster_name = cluster_details['name']
        admin_username = cluster_details['user']['admin_username']
        master_public_ip_address = cluster_details['master']['public_ip_address']
    except Exception:
        return

    copy_files_to_node(
        local_path=f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/details.yml",
        remote_dir=f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}",
        admin_username=admin_username, node_ip_address=master_public_ip_address
    )
    logger.debug("Sync cluster details with master")


def _sync_job_details(cluster_name: str, job_name: str, job_details: dict) -> None:
    try:
        cluster_details = load_cluster_details(cluster_name=cluster_name)
        admin_username = cluster_details['user']['admin_username']
        master_public_ip_address = cluster_details['master']['public_ip_address']
    except Exception:
        return

    copy_files_to_node(
        local_path=f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/jobs/{job_name}/details.yml",
        remote_dir=f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/jobs/{job_name}",
        admin_username=admin_username, node_ip_address=master_public_ip_address
    )
    logger.debug("Sync job details with master")
