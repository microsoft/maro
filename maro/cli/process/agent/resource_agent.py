# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import multiprocessing as mp
import os
import time

import redis

from maro.cli.utils.params import LocalParams
from maro.cli.utils.resource_executor import ResourceInfo
from maro.utils.exception.cli_exception import BadRequestError


class ResourceTrackingAgent(mp.Process):
    def __init__(
        self,
        check_interval: int = 60
    ):
        super().__init__()
        self.check_interval = check_interval
        self._redis_connection = redis.Redis(host="localhost", port=LocalParams.RESOURCE_REDIS_PORT)
        try:
            self._redis_connection.hset(
                LocalParams.RESOURCE_INFO,
                "agent_pid",
                os.getpid()
            )
        except Exception:
            raise BadRequestError(
                f"Failure to connect to Resource Redis."
                f"Please make sure at least one cluster running."
            )

        self._set_static_resource_info()

    def _set_static_resource_info(self):
        resource = ResourceInfo.get_static_info()
        self._redis_connection.hset(
            LocalParams.RESOURCE_INFO,
            "resource",
            json.dumps(resource)
        )

    def run(self) -> None:
        """Start tracking node status and updating details.

        Returns:
            None.
        """
        while True:
            start_time = time.time()
            self.push_local_resource_usage()
            time.sleep(max(self.check_interval - (time.time() - start_time), 0))

    def push_local_resource_usage(self):
        resource_usage = ResourceInfo.get_dynamic_info(self.check_interval)

        self._redis_connection.rpush(
            LocalParams.CPU_USAGE,
            json.dumps(resource_usage["cpu_usage_per_core"])
        )

        self._redis_connection.rpush(
            LocalParams.MEMORY_USAGE,
            json.dumps(resource_usage["memory_usage"])
        )

        self._redis_connection.rpush(
            LocalParams.GPU_USAGE,
            json.dumps(resource_usage["gpu_memory_usage"])
        )


if __name__ == "__main__":
    resource_agent = ResourceTrackingAgent()
    resource_agent.start()
