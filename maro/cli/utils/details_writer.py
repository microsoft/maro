# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

import yaml

from maro.cli.utils.params import GlobalPaths


class DetailsWriter:
    @staticmethod
    def save_cluster_details(cluster_name: str, cluster_details: dict) -> None:
        os.makedirs(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}", exist_ok=True)
        with open(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}/details.yml", "w") as fw:
            yaml.safe_dump(cluster_details, fw)

    @staticmethod
    def save_local_cluster_details(cluster_details: dict) -> dict:
        with open(file=f"{GlobalPaths.ABS_MARO_LOCAL_CLUSTER}/cluster_details.yml", mode="w") as fw:
            cluster_details = yaml.safe_dump(data=cluster_details, stream=fw)
        return cluster_details

    @staticmethod
    def save_local_node_details(node_details: dict) -> dict:
        with open(file=f"{GlobalPaths.ABS_MARO_LOCAL_CLUSTER}/node_details.yml", mode="w") as fw:
            node_details = yaml.safe_dump(data=node_details, stream=fw)
        return node_details

    @staticmethod
    def save_job_details(cluster_name: str, job_name: str, job_details: dict) -> None:
        with open(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}/jobs/{job_name}/details.yml", "w") as fw:
            yaml.safe_dump(job_details, fw)

    @staticmethod
    def save_schedule_details(cluster_name: str, schedule_name: str, schedule_details: dict) -> None:
        with open(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}/schedules/{schedule_name}/details.yml", "w") as fw:
            yaml.safe_dump(schedule_details, fw)
