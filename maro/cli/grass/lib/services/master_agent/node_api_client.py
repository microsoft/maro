# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import requests


class NodeApiClientV1:
    @staticmethod
    def stop_container(hostname: str, port: int, container_name: str) -> dict:
        response = requests.post(url=f"http://{hostname}:{port}/v1/containers/{container_name}:stop")
        return response.json()

    @staticmethod
    def remove_container(hostname: str, port: int, container_name: str) -> dict:
        response = requests.delete(url=f"http://{hostname}:{port}/v1/containers/{container_name}")
        return response.json()

    @staticmethod
    def create_container(hostname: str, port: int, create_config: dict) -> dict:
        response = requests.post(url=f"http://{hostname}:{port}/v1/containers", json=create_config)
        return response.json()
