# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import requests


class NodeApiClientV1:
    """Client class for Node API Server.
    """

    @staticmethod
    def create_container(node_hostname: str, node_api_server_port: int, create_config: dict) -> dict:
        response = requests.post(
            url=f"http://{node_hostname}:{node_api_server_port}/v1/containers",
            json=create_config
        )
        return response.json()

    @staticmethod
    def stop_container(node_hostname: str, node_api_server_port: int, container_name: str) -> dict:
        response = requests.post(
            url=f"http://{node_hostname}:{node_api_server_port}/v1/containers/{container_name}:stop"
        )
        return response.json()

    @staticmethod
    def remove_container(node_hostname: str, node_api_server_port: int, container_name: str) -> dict:
        response = requests.delete(
            url=f"http://{node_hostname}:{node_api_server_port}/v1/containers/{container_name}"
        )
        return response.json()
