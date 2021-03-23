# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import subprocess

import psutil
import redis

from maro.cli.process.utils.details import close_by_pid, get_redis_pid_by_port
from maro.cli.utils.params import LocalParams, LocalPaths
from maro.cli.utils.subprocess import Subprocess
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class ResourceInfo:
    @staticmethod
    def get_static_info() -> dict:
        """ Get static resource information about local environment.

        Returns:
            Tuple[int, list]: (total cpu number, [cpu usage per core])
        """
        static_info = {}
        static_info["cpu"] = psutil.cpu_count()

        memory = psutil.virtual_memory()
        static_info["total_memory"] = round(float(memory.total) / (1024 ** 2), 2)
        static_info["memory"] = round(float(memory.free) / (1024 ** 2), 2)

        gpu_static_command = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits"
        try:
            return_str = Subprocess.run(command=gpu_static_command)
            gpus_info = return_str.split(os.linesep)
            static_info["gpu"] = len(gpus_info) - 1  # (int) logical number
            static_info["gpu_name"] = []
            static_info["gpu_memory"] = []
            for info in gpus_info:
                name, total_memory = info.split(", ")
                static_info["gpu_name"].append(name)
                static_info["gpu_memory"].append(total_memory)
        except Exception:
            static_info["gpu"] = 0

        return static_info

    @staticmethod
    def get_dynamic_info(interval: int = None) -> dict:
        """ Get dynamic resource information about local environment.

        Returns:
            Tuple[float]: (total memory, free memory, used memory, memory usage)
        """
        dynamic_info = {}
        dynamic_info["cpu_usage_per_core"] = psutil.cpu_percent(interval=interval, percpu=True)

        memory = psutil.virtual_memory()
        dynamic_info["memory_usage"] = memory.percent / 100

        gpu_dynamic_command = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
        dynamic_info["gpu_memory_usage"] = []
        try:
            return_str = Subprocess.run(command=gpu_dynamic_command)
            memory_usage_per_gpu = return_str.split("\n")
            for single_usage in memory_usage_per_gpu:
                dynamic_info["gpu_memory_usage"].append(float(single_usage))
        except Exception:
            pass

        return dynamic_info


class LocalResourceExecutor:
    def __init__(self):
        self._redis_connection = redis.Redis(host="localhost", port=LocalParams.RESOURCE_REDIS_PORT)
        try:
            self._redis_connection.ping()
        except Exception:
            start_redis_command = f"redis-server --port {str(LocalParams.RESOURCE_REDIS_PORT)} --daemonize yes"
            _ = Subprocess.run(start_redis_command)

            # Start Resource Agents
            start_agent_command = f"python {LocalPaths.MARO_RESOURCE_AGENT}"
            _ = subprocess.Popen(start_agent_command, shell=True)

    def add_cluster(self):
        if self._redis_connection.hexists(LocalParams.RESOURCE_INFO, "connections"):
            current_connection = self._redis_connection.hget(
                LocalParams.RESOURCE_INFO,
                "connections"
            )
        else:
            current_connection = 0

        self._redis_connection.hset(LocalParams.RESOURCE_INFO, "connections", json.dumps(int(current_connection) + 1))

    def sub_cluster(self):
        current_connection = self._redis_connection.hget(
            LocalParams.RESOURCE_INFO,
            "connections"
        )

        self._redis_connection.hset(LocalParams.RESOURCE_INFO, "connections", json.dumps(int(current_connection) - 1))

        if int(current_connection) == 1:
            self._quit()

    def _quit(self):
        # Stop resource agents.
        agent_pid = self._redis_connection.hget(
            LocalParams.RESOURCE_INFO,
            "agent_pid"
        )
        close_by_pid(pid=int(agent_pid), recursive=True)
        logger.info("Resource agents exited!")

        # Close Resource Redis.
        redis_pid = get_redis_pid_by_port(LocalParams.RESOURCE_REDIS_PORT)
        close_by_pid(redis_pid, recursive=False)
        logger.info("Resource Redis exited!")

    def get_available_resource(self):
        if self._redis_connection.hexists(LocalParams.RESOURCE_INFO, "available_resource"):
            available_resource = json.loads(
                self._redis_connection.hget(LocalParams.RESOURCE_INFO, "available_resource")
            )
        else:
            available_resource = self.get_local_resource()

        return available_resource

    def set_available_resource(self, available_resource: dict):
        self._redis_connection.hset(
            LocalParams.RESOURCE_INFO,
            "available_resource",
            json.dumps(available_resource)
        )

    def get_local_resource(self):
        if self._redis_connection.hexists(LocalParams.RESOURCE_INFO, "resource"):
            static_resource = json.loads(
                self._redis_connection.hget(LocalParams.RESOURCE_INFO, "resource")
            )
        else:
            static_resource = ResourceInfo.get_static_info()

        return static_resource

    def get_local_resource_usage(self, previous_length: int):
        usage_dict = {}
        usage_dict["cpu"] = self._redis_connection.lrange(
            LocalParams.CPU_USAGE,
            previous_length, -1
        )
        usage_dict["memory"] = self._redis_connection.lrange(
            LocalParams.MEMORY_USAGE,
            previous_length, -1
        )
        usage_dict["gpu"] = self._redis_connection.lrange(
            LocalParams.GPU_USAGE,
            previous_length, -1
        )

        return usage_dict
