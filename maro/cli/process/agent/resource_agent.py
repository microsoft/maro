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
        check_interval: int = 30
    ):
        super().__init__()
        self._redis_connection = redis.Redis(host="localhost", port=LocalParams.RESOURCE_REDIS_PORT)
        try:
            if self._redis_connection.hexists(LocalParams.RESOURCE_INFO, "check_interval"):
                self._check_interval = int(self._redis_connection.hget(LocalParams.RESOURCE_INFO, "check_interval"))
            else:
                self._check_interval = check_interval
        except Exception:
            raise BadRequestError(
                "Failure to connect to Resource Redis."
                "Please make sure at least one cluster running."
            )

        self._set_resource_info()

    def _set_resource_info(self):
        # Set resource agent pid.
        self._redis_connection.hset(
            LocalParams.RESOURCE_INFO,
            "agent_pid",
            os.getpid()
        )

        # Set resource agent check interval.
        self._redis_connection.hset(
            LocalParams.RESOURCE_INFO,
            "check_interval",
            json.dumps(self._check_interval)
        )

        # Push static resource information into Redis.
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
            time.sleep(max(self._check_interval - (time.time() - start_time), 0))

            self._check_interval = int(self._redis_connection.hget(LocalParams.RESOURCE_INFO, "check_interval"))

    def push_local_resource_usage(self):
        resource_usage = ResourceInfo.get_dynamic_info(self._check_interval)

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
