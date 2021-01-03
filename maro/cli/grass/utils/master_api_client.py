# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import requests


class MasterApiClient:
    def __init__(self, master_ip_address: str, api_server_port: int):
        self.master_api_server_url_prefix = f"http://{master_ip_address}:{api_server_port}"

    # Master related.

    def get_master(self):
        response = requests.get(url=f"{self.master_api_server_url_prefix}/master")
        return response.json()

    def create_master(self, master_details: dict) -> dict:
        response = requests.post(url=f"{self.master_api_server_url_prefix}/master", json=master_details)
        return response.json()

    def delete_master(self):
        response = requests.delete(url=f"{self.master_api_server_url_prefix}/master")
        return response.json()

    # Nodes related.

    def list_nodes(self) -> list:
        response = requests.get(url=f"{self.master_api_server_url_prefix}/nodes")
        return response.json()

    def get_node(self, node_name: str) -> dict:
        response = requests.get(url=f"{self.master_api_server_url_prefix}/nodes/{node_name}")
        return response.json()

    def create_node(self, node_details: dict) -> dict:
        response = requests.post(url=f"{self.master_api_server_url_prefix}/nodes", json=node_details)
        return response.json()

    def delete_node(self, node_name: str) -> dict:
        response = requests.delete(url=f"{self.master_api_server_url_prefix}/nodes/{node_name}")
        return response.json()

    # Containers related.

    def list_containers(self):
        response = requests.get(url=f"{self.master_api_server_url_prefix}/containers")
        return response.json()

    # Jobs related.

    def list_jobs(self) -> list:
        response = requests.get(url=f"{self.master_api_server_url_prefix}/jobs")
        return response.json()

    def get_job(self, job_name: str) -> dict:
        response = requests.get(url=f"{self.master_api_server_url_prefix}/jobs/{job_name}")
        return response.json()

    def create_job(self, job_details: dict) -> dict:
        response = requests.post(url=f"{self.master_api_server_url_prefix}/jobs", json=job_details)
        return response.json()

    def delete_job(self, job_name: str) -> dict:
        response = requests.post(url=f"{self.master_api_server_url_prefix}/jobs/{job_name}")
        return response.json()

    def clean_jobs(self):
        response = requests.post(url=f"{self.master_api_server_url_prefix}/jobs:clean")
        return response.json()

    # Image files related.

    def list_image_files(self) -> list:
        response = requests.get(url=f"{self.master_api_server_url_prefix}/imageFiles")
        return response.json()

    def get_image_file(self, image_file_name: str) -> dict:
        response = requests.get(url=f"{self.master_api_server_url_prefix}/imageFiles/{image_file_name}")
        return response.json()

    def create_image_file(self, image_file_details: dict) -> dict:
        response = requests.post(url=f"{self.master_api_server_url_prefix}/imageFiles", json=image_file_details)
        return response.json()
