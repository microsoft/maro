import inquirer
import os
import subprocess
import logging
import yaml
import redis
import getpass
import json
import socket

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('logger')

def install_docker(delta_workers_info):
    admin_username = delta_workers_info['adminUsername']
    for worker in delta_workers_info["virtualMachines"]:
        install_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{worker['IP']} 'bash /code_point/bin/install_docker.sh'"
        res = subprocess.run(install_bin, shell=True, capture_output=True)
        
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logging.info(f"run {install_bin} success!")

def unpack_docker_images(delta_workers_info, image_name):
    admin_username = delta_workers_info['adminUsername']
    for worker in delta_workers_info["virtualMachines"]:
        load_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{worker['IP']} 'sudo docker load < /code_point/{image_name}'"
        res = subprocess.run(load_bin, shell=True, capture_output=True)
        
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logging.info(f"run {load_bin} success!")

def launch_job():
    if not os.path.exists('job_config'):
        raise Exception("please generate job config before allocate jobs!")
    
    host_name = socket.gethostname()

    questions = [
        inquirer.List(
            'jobGroupName', 
            message="Which is the job group name you want to launch?",
            choices=os.listdir("job_config/"),
            carousel=True,
        ),
        inquirer.Text(
            'imgName',
            message="What is the name of the image you want to use?",
            default="maro/ecr/cpu/latest",
        ),
        inquirer.Text(
            'componentPath',
            message="where is the component python file directory in docker?",
            default="/maro_dist/examples/ecr/q_learning/distributed_mode",
        )
    ]

    if host_name != "god":
        questions += [
            inquirer.List(
                'resourceGroupName',
                message="Which is the job group name you want to launch?",
                choices=[resource_group_name[:-5] for resource_group_name in os.listdir("azure_template/resource_group_info")],
                carousel=True,
            ),
            inquirer.Text(
                'codebaseName',
                message="What is the name of your codebase?",
            ),
        ]

    answers = inquirer.prompt(questions)
    job_group_name = answers['jobGroupName']
    img_name = answers['imgName']
    component_path = answers['componentPath']

    if host_name != "god":
        resource_group_name = answers['resourceGroupName']
        codebase_name = answers['codebaseName']
        with open(f'azure_template/resource_group_info/{resource_group_name}.json', 'r') as infile:
            exist_workers_info = json.load(infile)
    else:
        resource_group_name = None
        codebase_name = os.getcwd().split("/")[2]
        with open(f'/home/{getpass.getuser()}/resource_group_info.json', 'r') as infile:
            exist_workers_info = json.load(infile)

    require_resources = dict()
    for job_config in os.listdir(f"job_config/{job_group_name}"):
        with open(f"job_config/{job_group_name}/{job_config}", 'r') as infile:
            config = yaml.safe_load(infile)
        job_name = job_config[:-4]
        require_resources[job_name] = config['resources']
    
    # available_resources = get_available_resources(resource_group_name)

    # allocate_plan = best_fit_allocate(require_resources, available_resources)

    allocate_plan = {
        "environment_runner.0" : "worker0",
        "learner.0" : "worker1",
        "learner.1" : "worker1",
        "learner.2" : "worker1",
        "learner.3" : "worker1",
        "learner.4" : "worker1",
    }
    
    admin_username = exist_workers_info['adminUsername']
    worker_ip_dict = dict()
    
    for worker in exist_workers_info['virtualMachines']:
        if worker['name'] != "god":
            worker_ip_dict[worker['name']] = worker['IP']
    
    for job_name, worker_name in allocate_plan.items():
        envopt = f"-e CONFIG=/maro_dist/tools/azure_orch/job_config/{job_group_name}/{job_name}.yml"
        component_type = job_name.split(".")[0]
        cmd = f"python3 {component_path}/{component_type}.py"

        job_launch_bin = f"sudo docker run -it -d --name {job_group_name}_{job_name} --network host -v /code_point/{codebase_name}/maro/:/maro_dist {envopt} {img_name} {cmd}"
        ssh_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{worker_ip_dict[worker_name]} '{job_launch_bin}'"

        res = subprocess.run(ssh_bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info(f"run {ssh_bin} success!")

def best_fit_allocate(require_resources, available_resources):
    allocate_plan = dict()

    for job_name, requirement in require_resources.items():
        best_fit_worker = ""
        best_fit_score = 0
        score = 0
        for worker_name, free in free_resources.itmes():
            if requirement['CPU_cores'] / free['CPU_cores'] <= 1 and requirement['GPU_mem'] / free['GPU_mem'] <= 1 and requirement['mem'] / free['mem'] <= 1:
                score = requirement['CPU_cores'] / free['CPU_cores'] + \
                            requirement['GPU_mem'] / free['GPU_mem'] + \
                                requirement['mem'] / free['mem']
                if score > best_fit_score:
                    best_fit_score = score
                    best_fit_worker = worker_name
        if best_fit_worker == "":
            logging.error(f"can not allocate job [{job_name}] due to resouce limitation")
            raise("!!!")
        else:
            allocate_plan[job_name] = best_fit_worker
    
    logging.info(f"allocate plan: {allocate_plan}")

    return allocate_plan

def get_available_resources(resource_group_name=None):
    if resource_group_name:
        pass
    else:
        redis_connection = redis.StrictRedis(host="localhost", port="6379")
        free_resources = redis_connection.hgetall('resources')

    return free_resources

