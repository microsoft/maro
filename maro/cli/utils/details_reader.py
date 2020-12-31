# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import yaml

from maro.cli.utils.params import GlobalPaths


class DetailsReader:
    @staticmethod
    def load_cluster_details(cluster_name: str) -> dict:
        with open(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}/details.yml", "r") as fr:
            cluster_details = yaml.safe_load(fr)
        return cluster_details

    @staticmethod
    def load_job_details(cluster_name: str, job_name: str) -> dict:
        with open(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}/jobs/{job_name}/details.yml", "r") as fr:
            details = yaml.safe_load(fr)
        return details

    @staticmethod
    def load_schedule_details(cluster_name: str, schedule_name: str) -> dict:
        with open(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}/schedules/{schedule_name}/details.yml", "r") as fr:
            details = yaml.safe_load(fr)
        return details
