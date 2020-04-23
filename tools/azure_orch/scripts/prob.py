# from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
# from psutil import cpu_percent, cpu_count, virtual_memory
# from socket import gethostname
import redis
# import os
import time
import subprocess
import json

class Prob():
    def __init__(self):
        self._redis_connection = redis.StrictRedis(host='localhost', port='6379')

    def update_resources(self):
        while True:
            # nvmlInit()
            # handle = nvmlDeviceGetHandleByIndex(0)
            # meminfo = nvmlDeviceGetMemoryInfo(handle)
            # free_GPU_mem = meminfo.free / (1024 * 1024)

            # mem = virtual_memory()
            # free_mem = (mem.total - mem.used) / (1024 * 1024)

            # free_CPU_cores = (100 - cpu_percent()) / 100 * cpu_count()

            with open("/resource_group_info.json", "r") as infile:
                resource_group_info = json.load(infile)

            admin_username = resource_group_info['adminUsername']
            for worker in resource_group_info['virtualMachines']:
                if worker['name'] != "god":
                    ip_addr = worker['IP']
                    
                    res = subprocess.run(f"ssh -o StrictHostKeyChecking=no {admin_username}@{ip_addr} bash /code_point/bin/prob_resource.sh", shell=True, capture_output=True)
                    if res.returncode:
                        raise Exception(res.stderr)
                    else:
                        resources_info = str(res.stdout, 'utf-8').split(",")

                    free_resources = {"free_GPU_mem" : 999999,
                                    "free_mem": int(resources_info[0]) / 1024,
                                    "free_CPU_cores": int(resources_info[1]) * int(resources_info[2]) / 100}
                    
                    # print(free_resources)

                    self._redis_connection.hset("resources", gethostname(), json.dumps(free_resources))

            time.sleep(5)

if __name__ == "__main__":
    prob = Prob()
    prob.update_resources()