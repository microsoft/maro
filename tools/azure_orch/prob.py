from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from psutil import cpu_percent, cpu_count, virtual_memory
from socket import gethostname
import redis
import os
import time

import json

class Prob():
    def __init__(self):
        self._redis_connection = redis.StrictRedis(host=os.environ['REDIS_ADDRESS'], port=os.environ['REDIS_PORT'])

    def update_resources(self):
        while True:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            meminfo = nvmlDeviceGetMemoryInfo(handle)
            free_GPU_mem = meminfo.free / (1024 * 1024)

            mem = virtual_memory()
            free_mem = (mem.total - mem.used) / (1024 * 1024)

            free_CPU_cores = (100 - cpu_percent()) / 100 * cpu_count()

            free_resources = {"free_GPU_mem" : free_GPU_mem,
                            "free_mem": free_mem,
                            "free_CPU_cores": free_CPU_cores}
            
            # print(free_resources)

            self._redis_connection.hset("resources", gethostname(), json.dumps(free_resources))

            time.sleep(5)

prob = Prob()
prob.update_resources()