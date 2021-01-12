# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

import yaml


class DetailsReader:
    @staticmethod
    def load_cluster_details(cluster_name: str) -> dict:
        with open(os.path.expanduser(f"~/.maro-shared/clusters/{cluster_name}/details.yml"), "r") as fr:
            cluster_details = yaml.safe_load(fr)
        return cluster_details

    @staticmethod
    def load_local_cluster_details() -> dict:
        with open(file=os.path.expanduser(f"~/.maro-local/cluster/cluster_details.yml"), mode="r") as fr:
            cluster_details = yaml.safe_load(stream=fr)
        return cluster_details

    @staticmethod
    def load_local_node_details() -> dict:
        with open(file=os.path.expanduser(f"~/.maro-local/cluster/node_details.yml"), mode="r") as fr:
            node_details = yaml.safe_load(stream=fr)
        return node_details
