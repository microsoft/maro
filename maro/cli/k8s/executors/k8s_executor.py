# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
from abc import abstractmethod, ABC

import yaml
from kubernetes import client

from maro.cli.utils.params import GlobalPaths
from maro.cli.utils.subprocess import SubProcess
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class K8sExecutor(ABC):
    def __init__(self, cluster_details: dict):
        self.cluster_details = cluster_details

        # General configs
        self.cluster_name = self.cluster_details["name"]
        self.cluster_id = self.cluster_details["id"]

        # Init k8s_client env
        self.load_k8s_context()

    # Create related.

    @staticmethod
    def _init_redis():
        k8s_client = client.AppsV1Api()
        with open(f"{GlobalPaths.ABS_MARO_K8S_LIB}/configs/redis/redis.yml", "r") as fr:
            redis_deployment = yaml.safe_load(fr)
        k8s_client.create_namespaced_deployment(body=redis_deployment, namespace="default")

    @staticmethod
    @abstractmethod
    def _init_nvidia_plugin():
        """ Init nvidia plugin for K8s Cluster.

        Different providers may have different loading mechanisms.

        Returns:
            None.
        """
        pass

    # Job related.

    @staticmethod
    def list_job():
        # Get jobs details
        command = "kubectl get jobs -o=json"
        return_str = SubProcess.run(command)
        job_details_list = json.loads(return_str)["items"]
        jobs_details = {}
        for job_details in job_details_list:
            jobs_details[job_details["metadata"]["labels"]["jobName"]] = job_details

        # Print details
        logger.info(
            json.dumps(
                jobs_details,
                indent=4, sort_keys=True
            )
        )

    # Utils related

    @abstractmethod
    def load_k8s_context(self):
        """ Load k8s context of the MARO cluster.

        Different providers have different loading mechanisms,
        but every override methods must invoke "config.load_kube_config()" at the very end.

        Returns:
            None.
        """
        pass
