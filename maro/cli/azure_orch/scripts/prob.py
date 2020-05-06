import redis
import time
import subprocess
import json
import getpass

class Prob():
    def __init__(self):
        self._redis_connection = redis.StrictRedis(host='localhost', port='6379')

    def update_resources(self):
        while True:

            with open(f"/home/{getpass.getuser()}/resource_group_info.json", "r") as infile:
                resource_group_info = json.load(infile)

            admin_username = resource_group_info['admin_username']
            for node in resource_group_info['virtual_machines']:
                if node['name'] != "god":
                    ip_addr = node['IP']
                    
                    res = subprocess.run(f"ssh -o StrictHostKeyChecking=no {admin_username}@{ip_addr} bash /code_repo/bin/prob_resource.sh", shell=True, capture_output=True)
                    if res.returncode:
                        free_resources = {
                            "free_GPU_mem": 0,
                            "free_mem": 0,
                            "free_CPU_cores": 0,
                        }
                    else:
                        resources_info = str(res.stdout, 'utf-8').split(",")
                        free_resources = {
                            "free_GPU_mem" : 999999,    # currently GPU mem not supported, so it is set to 99999
                            "free_mem": int(resources_info[0]) / 1024,
                            "free_CPU_cores": float(resources_info[1]) * int(resources_info[2]) / 100
                        }
                    
                    print(f"{node['name']}: {free_resources}")

                    self._redis_connection.hset("resources", node['name'], json.dumps(free_resources))

            time.sleep(5)

if __name__ == "__main__":
    prob = Prob()
    prob.update_resources()