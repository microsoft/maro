import inquirer
import os
import subprocess
import logging
import yaml
import redis

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def install_docker(delta_cluster_info):
    for worker in delta_cluster_info["virtualMachines"]:
        install_bin = f"ssh -o StrictHostKeyChecking=no {worker['IP']} 'bash /codepoint/tools/azure_orch/bin/install_docker.sh'"
        res = subprocess.run(install_bin, shell=True, capture_output=True)
        
        if res.returncode:
            logging.error(f"run {install_bin} failed!")
            raise("!!!")
        else:
            logging.info(f"run {install_bin} success!")


def build_cpu_docker_images(delta_cluster_info, image_name):
    admin_username = delta_cluster_info['adminUsername']
    #can be convert to multi-thread
    for worker in delta_cluster_info["virtualMachines"]:
        build_bin = f"sudo DOCKER_FILE=/codepoint/docker_files/cpu.dist.df DOCKER_FILE_DIR=/codepoint DOCKER_IMAGE_NAME={image_name} bash /codepoint/tools/azure_orch/bin/build_image.sh"
        ssh_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{worker['IP']} '{build_bin}'"
        res = subprocess.run(ssh_bin, shell=True)

        if res.returncode:
            logging.error(f"run {build_bin} failed! msg err: {res.stderr}")
            raise("!!!")
        else:
            logging.info(f"run {build_bin} success!")

def allocate_job():
    if not os.path.exists('job_config'):
        logging.error("please generate job config before allocate jobs!")
        raise("!!!")

    job_group_list = os.listdir(f"job_config/")

    questions = [
        inquirer.List(
            'jobGroupName', 
            message="Which is the job group name you want to launch?",
            choices=job_group_list,
            carousel=True,
        ),
        inquirer.Text(
            'imgName',
            mesage="What is the name of the image you want to use?",
            default="maro/ecr/cpu/latest",
        ),
        inquirer.Text(
            'componentPath',
            mesage="where is the component python file directory in docker?",
            default="/maro_dist/examples/ecr/q-learning/distributed_mode",
        )

    ]

    answers = inquirer.prompt(questions)
    job_group_name = answers['jobGroupName']
    img_name = answers['imgName']
    component_path = answers['componentPath']

    require_resources = dict()

    for job_config in os.listdir(f"job_config/{job_group_name}"):
        with open(f"job_config/{job_group_name}/{job_config}", 'r') as infile:
            config = yaml.safe_load(infile)
        job_name = job_config[:-4]
        require_resources[job_name] = config['resources']

    allocate_plan = best_fit_allocate(require_resources)

    for job_name, worker_name in allocate_plan.items():
        envopt = f"-e CONFIG=/maro_dist/tools/azure_orch/job_config/{job_group_name}/{job_name}.yml"
        component_type = job_name.split(".")[0]
        cmd = f"python3 {component_path}/{component_type}.py"
        job_launch_bin = f"docker run -it -d --name {job_group_name}_{job_name} --network host -v /codepoint:/maro_dist {envopt} {img_name} {cmd}"


def best_fit_allocate(require_resources):
    redis_connection = redis.StrictRedis(host="localhost", port="6379")

    free_resources = redis_connection.hgetall('resources')

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

if __name__ == "__main__":
    allocate_job()





    # ssh -o StrictHostKeyChecking=no tianyi@52.187.127.230 sudo docker run -it -d --name test_env --network host -v /codepoint:/maro_dist -e CONFIG=/maro_dist/tools/azure_orch/job_config/test_ep500_9NuJUi/environment_runner.0.yml maro/ecr/cpu/latest python3 /maro_dist/examples/ecr/q_learning/distributed_mode/environment_runner.py

    # ssh -o StrictHostKeyChecking=no tianyi@52.163.221.222 sudo docker run -it -d --name test_learner0 --network host -v /codepoint:/maro_dist -e CONFIG=/maro_dist/tools/azure_orch/job_config/test_ep500_9NuJUi/learner.0.yml maro/ecr/cpu/latest python3 /maro_dist/examples/ecr/q_learning/distributed_mode/learner.py
    # ssh -o StrictHostKeyChecking=no tianyi@52.163.221.222 sudo docker run -it -d --name test_learner1 --network host -v /codepoint:/maro_dist -e CONFIG=/maro_dist/tools/azure_orch/job_config/test_ep500_9NuJUi/learner.1.yml maro/ecr/cpu/latest python3 /maro_dist/examples/ecr/q_learning/distributed_mode/learner.py
    # ssh -o StrictHostKeyChecking=no tianyi@52.163.221.222 sudo docker run -it -d --name test_learner2 --network host -v /codepoint:/maro_dist -e CONFIG=/maro_dist/tools/azure_orch/job_config/test_ep500_9NuJUi/learner.2.yml maro/ecr/cpu/latest python3 /maro_dist/examples/ecr/q_learning/distributed_mode/learner.py
    # ssh -o StrictHostKeyChecking=no tianyi@52.163.221.222 sudo docker run -it -d --name test_learner3 --network host -v /codepoint:/maro_dist -e CONFIG=/maro_dist/tools/azure_orch/job_config/test_ep500_9NuJUi/learner.3.yml maro/ecr/cpu/latest python3 /maro_dist/examples/ecr/q_learning/distributed_mode/learner.py
    # ssh -o StrictHostKeyChecking=no tianyi@52.163.221.222 sudo docker run -it -d --name test_learner4 --network host -v /codepoint:/maro_dist -e CONFIG=/maro_dist/tools/azure_orch/job_config/test_ep500_9NuJUi/learner.4.yml maro/ecr/cpu/latest python3 /maro_dist/examples/ecr/q_learning/distributed_mode/learner.py 