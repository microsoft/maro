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

def install_docker(delta_nodes_info):
    admin_username = delta_nodes_info['admin_username']
    for node in delta_nodes_info["virtual_machines"]:
        install_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{node['IP']} 'bash /code_repo/bin/install_docker.sh'"
        res = subprocess.run(install_bin, shell=True, capture_output=True)
        
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logging.info(f"run {install_bin} success!")

def unpack_docker_images(delta_nodes_info, image_name):
    admin_username = delta_nodes_info['admin_username']
    for node in delta_nodes_info["virtual_machines"]:
        load_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{node['IP']} 'sudo docker load < /code_repo/{image_name}'"
        res = subprocess.run(load_bin, shell=True, capture_output=True)
        
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logging.info(f"run {load_bin} success!")

def launch_job():
    host_name = socket.gethostname()

    questions = [
        inquirer.Text(
            'branch_name',
            message="What is the name of your branch?",
        ),
    ]

    branch_name = inquirer.prompt(questions)['branch_name']

    job_config_path = "/maro/dist/job_config" if host_name != 'god' else f"/code_repo/{branch_name}/maro/job_config"
    if not os.path.exists(job_config_path):
        raise Exception("please generate job config before allocate jobs!")

    questions = [
        inquirer.List(
            'job_group_name', 
            message="Which is the job group name you want to launch?",
            choices=os.listdir(job_config_path),
            carousel=True,
        ),
        inquirer.Text(
            'img_name',
            message="What is the name of the image you want to use?",
            default="maro/ecr/cpu/latest",
        ),
        inquirer.Text(
            'component_path',
            message="where is the component python file directory in docker?",
            default="/maro_dist/examples/ecr/q_learning/distributed_mode",
        )
    ]

    if host_name != "god":
        questions += [
            inquirer.List(
                'resource_group_name',
                message="Which is the resource group name you want to launch?",
                choices=[resource_group_name[:-5] for resource_group_name in os.listdir("/maro/dist/azure_template/resource_group_info")],
                carousel=True,
            ),
        ]

    answers = inquirer.prompt(questions)
    job_group_name = answers['job_group_name']
    img_name = answers['img_name']
    component_path = answers['component_path']

    if host_name != "god":
        resource_group_name = answers['resource_group_name']
        with open(f'/maro/dist/azure_template/resource_group_info/{resource_group_name}.json', 'r') as infile:
            resource_group_info = json.load(infile)
    else:
        resource_group_name = None
        with open(f'/home/{getpass.getuser()}/resource_group_info.json', 'r') as infile:
            resource_group_info = json.load(infile)

    require_resources = dict()
    for job_config in os.listdir(f"{job_config_path}/{job_group_name}"):
        with open(f"{job_config_path}/{job_group_name}/{job_config}", 'r') as infile:
            config = yaml.safe_load(infile)
        job_name = job_config[:-4]
        require_resources[job_name] = config['resources']
    
    # available_resources = get_available_resources(resource_group_name)

    # allocate_plan = allocate(require_resources, available_resources)

    allocate_plan = {
        "environment_runner.0" : "node0",
        "learner.0" : "node0",
        "learner.1" : "node0",
        "learner.2" : "node0",
        "learner.3" : "node0",
        "learner.4" : "node0",
    }
    
    admin_username = resource_group_info['admin_username']
    node_ip_dict = dict()
    
    for node in resource_group_info['virtual_machines']:
        if node['name'] != "god":
            node_ip_dict[node['name']] = node['IP']
    
    for job_name, node_name in allocate_plan.items():
        envopt = f"-e CONFIG=/maro_dist/job_config/{job_group_name}/{job_name}.yml"
        component_type = job_name.split(".")[0]
        cmd = f"python3 {component_path}/{component_type}.py"

        cpu_cores = require_resources[job_name]['CPU_cores']
        GPU_mem = require_resources[job_name]['GPU_mem']
        mem = require_resources[job_name]['mem']

        ssh_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{node_ip_dict[node_name]} "

        rm_container_bin = ssh_bin + f"'sudo docker ps -a | grep Exit | cut -d ' ' -f 1 | xargs sudo docker rm'"
        job_launch_bin = ssh_bin + f"'sudo docker run -it -d --cpus {cpu_cores} -m {mem}m --name {job_group_name}_{job_name} --network host -v /code_repo/{branch_name}/maro/:/maro_dist {envopt} {img_name} {cmd}'"
        # ssh_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{node_ip_dict[node_name]} '{job_launch_bin}'"

        subprocess.run(rm_container_bin, shell=True, capture_output=True)

        res = subprocess.run(job_launch_bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info(f"run {job_launch_bin} success!")

def allocate(require_resources, available_resources):
    allocate_plan = dict()

    for job_name, requirement in require_resources.items():
        best_fit_node = ""
        best_fit_score = 0
        score = 0
        for node_name, free in free_resources.itmes():
            if requirement['CPU_cores'] <= free['CPU_cores'] and \
                requirement['GPU_mem'] <= free['GPU_mem'] and \
                    requirement['mem'] <= free['mem']:
                score = requirement['CPU_cores'] / free['CPU_cores'] + \
                            requirement['GPU_mem'] / free['GPU_mem'] + \
                                requirement['mem'] / free['mem']
                if score > best_fit_score:
                    best_fit_score = score
                    best_fit_node = node_name
        if best_fit_node == "":
            raise Exception(f"can not allocate job [{job_name}] due to resouce limitation")
        else:
            allocate_plan[job_name] = best_fit_node
    
    logging.info(f"allocate plan: {allocate_plan}")

    return allocate_plan

def get_available_resources(resource_group_name=None):
    if resource_group_name:
        with open(f'/maro/dist/azure_template/resource_group_info/{resource_group_name}.json', 'r') as infile:
            resource_group_info = json.load(infile)

        god_IP = resource_group_info['virtual_machines'][0]['vnet_IP']

        redis_connection = redis.StrictRedis(host=god_IP, port="6379")
        free_resources = redis_connection.hgetall('resources')
    else:
        redis_connection = redis.StrictRedis(host="localhost", port="6379")
        free_resources = redis_connection.hgetall('resources')

    return free_resources

