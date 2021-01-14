# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json

from kubernetes import client


class K8sDetailsWriter:
    @staticmethod
    def save_job_details(job_details: dict) -> None:
        job_name = job_details["name"]

        k8s_client = client.CoreV1Api()
        k8s_client.create_namespaced_config_map(
            body=client.V1ConfigMap(
                metadata=client.V1ObjectMeta(
                    name=f"job.details-{job_name}",
                    namespace="default"
                ),
                data={
                    "encoded_data": json.dumps(job_details)
                }
            ),
            namespace="default"
        )

    @staticmethod
    def save_schedule_details(schedule_details: dict) -> None:
        schedule_name = schedule_details["name"]

        k8s_client = client.CoreV1Api()
        k8s_client.create_namespaced_config_map(
            body=client.V1ConfigMap(
                metadata=client.V1ObjectMeta(
                    name=f"schedule.details-{schedule_name}",
                    namespace="default"
                ),
                data={
                    "encoded_data": json.dumps(schedule_details)
                }
            ),
            namespace="default"
        )
