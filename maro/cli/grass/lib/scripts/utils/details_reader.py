# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import yaml

from .params import Paths


class DetailsReader:
    """Reader class for details.
    """

    @staticmethod
    def load_cluster_details(cluster_name: str) -> dict:
        with open(file=f"{Paths.ABS_MARO_SHARED}/clusters/{cluster_name}/cluster_details.yml", mode="r") as fr:
            cluster_details = yaml.safe_load(fr)
        return cluster_details

    @staticmethod
    def load_local_cluster_details() -> dict:
        with open(file=f"{Paths.ABS_MARO_LOCAL}/cluster/cluster_details.yml", mode="r") as fr:
            cluster_details = yaml.safe_load(stream=fr)
        return cluster_details

    @staticmethod
    def load_local_node_details() -> dict:
        with open(file=f"{Paths.ABS_MARO_LOCAL}/cluster/node_details.yml", mode="r") as fr:
            node_details = yaml.safe_load(stream=fr)
        return node_details
