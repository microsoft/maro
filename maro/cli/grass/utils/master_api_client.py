# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.grass.utils.encrypted_requests import EncryptedRequests


class MasterApiClientV1:
    """Client class for Master API Server.
    """

    def __init__(
        self,
        master_hostname: str, master_api_server_port: int,
        user_id: str,
        master_to_dev_encryption_private_key: str,
        dev_to_master_encryption_public_key: str,
        dev_to_master_signing_private_key: str
    ):
        self._master_api_server_url_prefix = f"http://{master_hostname}:{master_api_server_port}/v1"
        self._encrypted_requests = EncryptedRequests(
            user_id=user_id,
            master_to_dev_encryption_private_key=master_to_dev_encryption_private_key,
            dev_to_master_encryption_public_key=dev_to_master_encryption_public_key,
            dev_to_master_signing_private_key=dev_to_master_signing_private_key
        )

    # Cluster related.

    def create_cluster(self, cluster_details: dict) -> dict:
        return self._encrypted_requests.post(
            url=f"{self._master_api_server_url_prefix}/cluster",
            json_dict=cluster_details
        )

    # Master related.

    def get_master(self) -> dict:
        return self._encrypted_requests.get(url=f"{self._master_api_server_url_prefix}/master")

    def create_master(self, master_details: dict) -> dict:
        return self._encrypted_requests.post(
            url=f"{self._master_api_server_url_prefix}/master",
            json_dict=master_details
        )

    def delete_master(self) -> dict:
        return self._encrypted_requests.delete(url=f"{self._master_api_server_url_prefix}/master")

    # Nodes related.

    def list_nodes(self) -> list:
        return self._encrypted_requests.get(url=f"{self._master_api_server_url_prefix}/nodes")

    def get_node(self, node_name: str) -> dict:
        return self._encrypted_requests.get(url=f"{self._master_api_server_url_prefix}/nodes/{node_name}")

    def create_node(self, node_details: dict) -> dict:
        return self._encrypted_requests.post(
            url=f"{self._master_api_server_url_prefix}/nodes",
            json_dict=node_details
        )

    def delete_node(self, node_name: str) -> dict:
        return self._encrypted_requests.delete(url=f"{self._master_api_server_url_prefix}/nodes/{node_name}")

    def start_node(self, node_name: str) -> dict:
        return self._encrypted_requests.post(url=f"{self._master_api_server_url_prefix}/nodes/{node_name}:start")

    def stop_node(self, node_name: str) -> dict:
        return self._encrypted_requests.post(url=f"{self._master_api_server_url_prefix}/nodes/{node_name}:stop")

    # Containers related.

    def list_containers(self) -> list:
        return self._encrypted_requests.get(url=f"{self._master_api_server_url_prefix}/containers")

    # Jobs related.

    def list_jobs(self) -> list:
        return self._encrypted_requests.get(url=f"{self._master_api_server_url_prefix}/jobs")

    def get_job(self, job_name: str) -> dict:
        return self._encrypted_requests.get(url=f"{self._master_api_server_url_prefix}/jobs/{job_name}")

    def create_job(self, job_details: dict) -> dict:
        return self._encrypted_requests.post(
            url=f"{self._master_api_server_url_prefix}/jobs",
            json_dict=job_details
        )

    def delete_job(self, job_name: str) -> dict:
        return self._encrypted_requests.delete(url=f"{self._master_api_server_url_prefix}/jobs/{job_name}")

    def clean_jobs(self):
        return self._encrypted_requests.post(url=f"{self._master_api_server_url_prefix}/jobs:clean")

    # Schedules related

    def list_schedules(self) -> list:
        return self._encrypted_requests.get(url=f"{self._master_api_server_url_prefix}/schedules")

    def get_schedule(self, schedule_name: str) -> dict:
        return self._encrypted_requests.get(url=f"{self._master_api_server_url_prefix}/schedules/{schedule_name}")

    def create_schedule(self, schedule_details: dict) -> dict:
        return self._encrypted_requests.post(
            url=f"{self._master_api_server_url_prefix}/schedules",
            json_dict=schedule_details
        )

    def stop_schedule(self, schedule_name: str) -> dict:
        return self._encrypted_requests.post(url=f"{self._master_api_server_url_prefix}/schedules/{schedule_name}:stop")

    # Image files related.

    def list_image_files(self) -> list:
        return self._encrypted_requests.get(url=f"{self._master_api_server_url_prefix}/imageFiles")

    def get_image_file(self, image_file_name: str) -> dict:
        return self._encrypted_requests.get(url=f"{self._master_api_server_url_prefix}/imageFiles/{image_file_name}")

    def create_image_file(self, image_file_details: dict) -> dict:
        return self._encrypted_requests.post(
            url=f"{self._master_api_server_url_prefix}/imageFiles",
            json_dict=image_file_details
        )

    # Visible related.
    def get_static_resource_info(self):
        return self._encrypted_requests.get(
            url=f"{self._master_api_server_url_prefix}/visible/static"
        )

    def get_dynamic_resource_info(self, previous_length: int):
        return self._encrypted_requests.get(
            url=f"{self._master_api_server_url_prefix}/visible/dynamic/{previous_length}"
        )

    def get_job_queue(self):
        return self._encrypted_requests.get(url=f"{self._master_api_server_url_prefix}/jobs/queue")
