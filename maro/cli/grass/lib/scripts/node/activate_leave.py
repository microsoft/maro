# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

import requests
import yaml


class Paths:
    MARO_LOCAL = "~/.maro-local"

    ABS_MARO_LOCAL = os.path.expanduser(MARO_LOCAL)


class DetailsReader:
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


class MasterApiClientV1:
    def __init__(self, master_hostname: str, master_api_server_port: int):
        self.master_api_server_url_prefix = f"http://{master_hostname}:{master_api_server_port}/v1"

    # Node related.

    def delete_node(self, node_name: str) -> int:
        response = requests.delete(url=f"{self.master_api_server_url_prefix}/nodes/{node_name}")
        return response.json()


if __name__ == '__main__':
    local_cluster_details = DetailsReader.load_local_cluster_details()
    local_node_details = DetailsReader.load_local_node_details()

    master_api_client = MasterApiClientV1(
        master_hostname=local_cluster_details["master"]["hostname"],
        master_api_server_port=local_cluster_details["master"]["api_server"]["port"]
    )
    master_api_client.delete_node(local_node_details["name"])
