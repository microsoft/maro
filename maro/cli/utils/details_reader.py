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
