# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple

import GPUtil
import psutil


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


_CPU_INFO = namedtuple("CPU_INFO", ["cpu_count", "cpu_usage_per_core"])
_MEMORY_INFO = namedtuple("MEMORY_INFO", ["total_memory", "free_memory", "used_memory", "memory_usage"])
_GPU_INFO = namedtuple("GPU_INFO", ["id", "name", "total_memory", "free_memory", "used_memory", "memory_usage"])


class ResourceInfo:
    @staticmethod
    def cpu_info(interval: int = None) -> tuple:
        """ Get CPU information about local environment.

        Returns:
            Tuple[int, list]: (total cpu number, [cpu usage per core])
        """
        cpu = psutil.cpu_count()
        cpu_usage_per_core = psutil.cpu_percent(interval=interval, percpu=True)
        return _CPU_INFO(cpu_count=cpu, cpu_usage_per_core=cpu_usage_per_core)

    @staticmethod
    def memory_info() -> tuple:
        """ Get memory information about local environment.

        Returns:
            Tuple[float]: (total memory, free memory, used memory, memory usage)
        """
        memory = psutil.virtual_memory()
        # Unit MB
        total_memory = round(float(memory.total) / (1024 ** 2), 2)
        free_memory = round(float(memory.free) / (1024 ** 2), 2)
        used_memory = round(float(memory.used) / (1024 ** 2), 2)
        memory_usage = memory.percent / 100

        return _MEMORY_INFO(
            total_memory=total_memory,
            free_memory=free_memory,
            used_memory=used_memory,
            memory_usage=memory_usage
        )

    @staticmethod
    def gpu_info() -> tuple:
        """ Get GPU information about local environment.

        Returns:
            Tuple: (GPU ID, GPU name, total memory, free memory, used memory, memory usage)
        """
        gpu_list = GPUtil.getGPUs()
        gpu_info = []
        if not gpu_list:
            return gpu_info

        for gpu in gpu_list:
            gpu_info.append(_GPU_INFO(
                id=gpu.id,
                name=gpu.name,
                total_memory=gpu.memoryTotal,
                free_memory=gpu.memoryFree,
                used_memory=gpu.memoryUsed,
                memory_usage=gpu.memoryUtil
            ))

        return gpu_info
