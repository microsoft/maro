# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json

from kubernetes import client


class K8sDetailsReader:
    @staticmethod
    def load_job_details(job_name: str) -> dict:
        k8s_client = client.CoreV1Api()
        config_map = k8s_client.read_namespaced_config_map(name=f"job.details-{job_name}", namespace="default")
        return json.loads(config_map.data["encoded_data"])

    @staticmethod
    def load_schedule_details(schedule_name: str) -> dict:
        k8s_client = client.CoreV1Api()
        config_map = k8s_client.read_namespaced_config_map(
            name=f"schedule.details-{schedule_name}",
            namespace="default"
        )
        return json.loads(config_map.data["encoded_data"])
