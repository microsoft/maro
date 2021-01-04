# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json

from maro.cli.utils.params import GlobalPaths
from maro.cli.utils.subprocess import SubProcess
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class K8sExecutor:

    # Create related.

    @staticmethod
    def _init_redis():
        # Apply k8s config
        command = f"kubectl apply -f {GlobalPaths.ABS_MARO_K8S_LIB}/configs/redis/redis.yml"
        _ = SubProcess.run(command)

    @staticmethod
    def _init_nvidia_plugin():
        # Create plugin namespace
        command = "kubectl create namespace gpu-resources"
        _ = SubProcess.run(command)

        # Apply k8s config
        command = f"kubectl apply -f {GlobalPaths.ABS_MARO_K8S_LIB}/configs/nvidia/nvidia-device-plugin.yml"
        _ = SubProcess.run(command)

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
