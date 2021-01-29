# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess

import psutil

from .subprocess import Subprocess


class BasicResource:
    """An abstraction class for computing resources.
    """

    def __init__(self, cpu: float, memory: float, gpu: float):
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu

    def __cmp__(self, other):
        if self.cpu == other.cpu and self.memory == other.memory and self.gpu == other.gpu:
            return 0
        elif self.cpu >= other.cpu and self.memory >= other.memory and self.gpu >= other.gpu:
            return 1
        else:
            return -1

    def __lt__(self, other):
        return self.__cmp__(other=other) == -1

    def __le__(self, other):
        return self.__cmp__(other=other) <= 0

    def __eq__(self, other):
        return self.__cmp__(other=other) == 0

    def __ne__(self, other):
        return self.__cmp__(other=other) != 0

    def __gt__(self, other):
        return self.__cmp__(other=other) == 1

    def __ge__(self, other):
        return self.__cmp__(other=other) >= 0


class ContainerResource(BasicResource):
    def __init__(self, container_name: str, cpu: float, memory: float, gpu: float):
        super().__init__(cpu=cpu, memory=memory, gpu=gpu)
        self.container_name = container_name


class NodeResource(BasicResource):
    def __init__(self, node_name: str, cpu: float, memory: float, gpu: float):
        super().__init__(cpu=cpu, memory=memory, gpu=gpu)
        self.node_name = node_name


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
            return_str = Subprocess.run(command=GET_GPU_INFO_COMMAND)
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
            return_str = Subprocess.run(command=GET_UTILIZATION_GPUS_COMMAND)
            memory_usage_per_gpu = return_str.split("\n")
            for single_usage in memory_usage_per_gpu:
                dynamic_info["gpu_memory_usage"].append(float(single_usage))
        except Exception:
            pass

        return dynamic_info
