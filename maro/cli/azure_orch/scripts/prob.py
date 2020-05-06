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

            admin_username = resource_group_info['admin_user_name']
            for node in resource_group_info['virtual_machines']:
                if node['name'] != "god":
                    ip_addr = node['IP']
                    
                    res = subprocess.run(f"ssh -o StrictHostKeyChecking=no {admin_username}@{ip_addr} bash /code_repo/bin/prob_resource.sh", shell=True, capture_output=True)
                    if res.returncode:
                        raise Exception(res.stderr)
                    else:
                        resources_info = str(res.stdout, 'utf-8').split(",")

                    free_resources = {"free_GPU_mem" : 999999,
                                    "free_mem": int(resources_info[0]) / 1024,
                                    "free_CPU_cores": float(resources_info[1]) * int(resources_info[2]) / 100}

                    self._redis_connection.hset("resources", node['name'], json.dumps(free_resources))

            time.sleep(5)

if __name__ == "__main__":
    prob = Prob()
    prob.update_resources()