# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

import yaml


class DetailsReader:
    @staticmethod
    def load_cluster_details(cluster_name: str) -> dict:
        with open(os.path.expanduser(f"~/.maro/clusters/{cluster_name}/details.yml"), "r") as fr:
            cluster_details = yaml.safe_load(fr)
        return cluster_details
