import getpass, os, subprocess
import logging
import inquirer
import json
import socket

import chalk
from tqdm import tqdm

from tools.azure_orch.scripts.gen_job_config import gen_job_config
from tools.azure_orch.scripts.docker_new import install_docker


logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('logger')

def create_god():
    resource_group_info = inquirer_resource_group()
    resource_group_name = resource_group_info['virtualMachineRG']

    if not os.path.exists('azure_template/resource_group_info'):
        os.mkdir('azure_template/resource_group_info')

    with open(f'azure_template/resource_group_info/{resource_group_name}.json', 'w') as outfile:  
        json.dump(resource_group_info, outfile, indent=4)
    
    config_bin = f"python azure_template/vmconfig.py azure_template/resource_group_info/{resource_group_name}.json azure_template/{resource_group_name} azure_template/parameters.json"
    
    create_bin = f"./azure_template/deploy.sh -i {resource_group_info['subscription']} " + \
                 f"-g {resource_group_info['virtualMachineRG']} -n DEFAULT " + \
                 f"-l {resource_group_info['location']} " + \
                 f"-t azure_template/template.json " + \
                 f"-p azure_template/{resource_group_name}/god.json"

    get_IP_bin = f"az network public-ip list -g {resource_group_info['virtualMachineRG']}"

    for bin in [config_bin, create_bin, get_IP_bin]:
        res = subprocess.run(bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            if bin[0] == 'a':
                ip_info = json.loads(res.stdout)
                god_ip = ip_info[0]["ipAddress"]
                
                get_vnetIP = "ip addr | grep 'state UP' -A2 | tail -n1 | awk '{print $2}' | cut -f1 -d '/'"
                res = subprocess.run(f"ssh -o StrictHostKeyChecking=no {resource_group_info['adminUsername']}@{god_ip} {get_vnetIP}", shell=True, capture_output=True)
                if res.returncode:
                    raise Exception(res.stderr)
                else:
                    god_vnet_IP = str(res.stdout, 'utf-8')

                resource_group_info['virtualMachines'][0]['IP'] = god_ip
                resource_group_info['virtualMachines'][0]['vnetIP'] = god_vnet_IP[:-1]
                logger.info("Now you can remote login your god machine. ")
                logger.info(chalk.green(f"COPY AND RUN: ssh {resource_group_info['adminUsername']}@{god_ip}"))

            else:
                logger.info(f"run {bin} sucess.")

    with open(f'azure_template/resource_group_info/{resource_group_name}.json', 'w') as outfile:
        json.dump(resource_group_info, outfile, indent=4)
    
    res = subprocess.run(f"scp -o StrictHostKeyChecking=no azure_template/resource_group_info/{resource_group_name}.json {resource_group_info['adminUsername']}@{god_ip}:~/resource_group_info.json", shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)

    return resource_group_name, resource_group_info

def inquirer_resource_group():
    res = subprocess.run("az account list", shell=True, capture_output=True)
    if res.stderr:
        res = subprocess.run("az login", shell=True, stdout=subprocess.PIPE)
    
    if res.returncode:
        raise Exception(res.stderr)
    
    subscription_id_list = []
    for subscription in json.loads(res.stdout):
        subscription_id_list += [f"{subscription['name']}: {subscription['id']}"]

    questions = [
        inquirer.Text(
            'adminUsername', 
            message="Who is the admin user on god?",
        ),
        inquirer.Text(
            'adminPublicKey',
            message="What is your public key? (default is ~/.ssh/id_rsa.pub)",
            default=open(f'/home/{getpass.getuser()}/.ssh/id_rsa.pub').read()
        ),
        inquirer.List(
            'subscription', 
            message="Which is your subscription?",
            choices=subscription_id_list,
            carousel=True,
        ),
        inquirer.Text(
            'location',
            message="In which location you will set up the god?",
            default="southeastasia"
        ),
        inquirer.Text(
            'virtualMachineRG',
            message="What name is your group?",
            default="maro_dist"
        ),
        inquirer.Text(
            'godSize',
            message="Which size is your god machine?",
            default="Standard_D16s_v3"
        ),

    ]

    resource_group_info = inquirer.prompt(questions)

    resource_group_info = {
        "adminPublicKey": resource_group_info["adminPublicKey"],
        "adminUsername": resource_group_info["adminUsername"],
        "virtualMachineRG": resource_group_info["virtualMachineRG"],
        "subscription": resource_group_info["subscription"].split(": ")[1],
        "location": resource_group_info["location"],
        "virtualMachines": [{
            "name": "god",
            "size": resource_group_info["godSize"]
        }]
    }

    return resource_group_info


def init_god(resource_group_info, image_name="maro/ecr/cpu/latest"):
    admin_username = resource_group_info["adminUsername"]
    god_ip = resource_group_info["virtualMachines"][0]["IP"]
    ssh_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{god_ip} "

    # gen pub key
    gen_pubkey_bin = ssh_bin + '''"ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa <<< y"'''
    get_pubkey_bin = ssh_bin + "cat ~/.ssh/id_rsa.pub"
    for bin in [gen_pubkey_bin, get_pubkey_bin]:
        res = subprocess.run(bin, shell=True, capture_output=True)
        if res.returncode:
            print(bin)
            raise Exception(res.stderr)
        else:
            if "cat" in bin:
                god_public_key = str(res.stdout, 'utf-8') 

    # scp necessary files
    mkdir_bin = ssh_bin + "sudo mkdir /code_point"
    chmod_bin = ssh_bin + "sudo chmod 777 /code_point"
    scp_bash_bin = f"scp -o StrictHostKeyChecking=no -r bin {admin_username}@{god_ip}:/code_point/"
    scp_df_bin = f"scp -o StrictHostKeyChecking=no -r ../../docker_files {admin_username}@{god_ip}:/code_point/"
    scp_requirements_bin = f"scp -o StrictHostKeyChecking=no ../../requirements.dev.txt {admin_username}@{god_ip}:/code_point/"
    scp_prob_bin = f"scp -o StrictHostKeyChecking=no scripts/prob.py {admin_username}@{god_ip}:/code_point/"
    for bin in [mkdir_bin, chmod_bin, scp_bash_bin, scp_df_bin, scp_requirements_bin, scp_prob_bin]:
        res = subprocess.run(bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info(f"run {bin} success!")

    # setup samba file server
    setup_samba_bin = ssh_bin + "sudo bash /code_point/bin/launch_samba.sh"
    res = subprocess.run(setup_samba_bin, shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)
    else:
        logger.info(f"setup samba file server on god success!")
    
    set_password_bin = ssh_bin + f"sudo smbpasswd -a {admin_username}"
    proc = subprocess.Popen(set_password_bin, shell=True, stdin=subprocess.PIPE)
    proc.communicate(input=b'maro_dist\nmaro_dist')

    # install docker
    install_docker_bin = ssh_bin + f"bash /code_point/bin/install_docker.sh"

    # build docker images
    build_docker_images_bin = ssh_bin + f"sudo DOCKER_FILE=/code_point/docker_files/cpu.dist.df DOCKER_FILE_DIR=/code_point DOCKER_IMAGE_NAME={image_name} bash /code_point/bin/build_image.sh"

    # save docker images to code point
    #TODO: support more docker files, change docker_image
    save_image_bin = ssh_bin + f"sudo docker save {image_name} > /code_point/docker_image"

    for bin in [install_docker_bin, build_docker_images_bin, save_image_bin]:
        res = subprocess.run(bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info(f"run {bin} success!")
    
    # launch redis
    launch_redis_bin = ssh_bin + "sudo bash /code_point/bin/launch_redis.sh"
    res = subprocess.run(launch_redis_bin, shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)
    else:
        logger.info(f"run {launch_redis_bin} success!")

    # prob resources
    # prob_resources(delta_workers_info, image_name)

def create_workers(resource_group_name=None):
    delta_workers_info = inquirer_workers(resource_group_name)

    delta_workers_info_name = f"{delta_workers_info['virtualMachineRG']}.delta_info"
    resource_group_name = delta_workers_info['virtualMachineRG']
    
    with open(f'azure_template/{delta_workers_info_name}.json', 'w') as outfile:  
        json.dump(delta_workers_info, outfile, indent=4)

    config_bin = f"python azure_template/vmconfig.py azure_template/{delta_workers_info_name}.json azure_template/{resource_group_name} azure_template/parameters.json"
    
    res = subprocess.run(config_bin, shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)
    else:
        logging.info(f"run {config_bin} sucess.")

    pbar = tqdm(total=len(delta_workers_info['virtualMachines']))

    for worker in delta_workers_info['virtualMachines']:
        create_bin = f"./azure_template/deploy.sh -i {delta_workers_info['subscription']} " + \
                     f"-g {delta_workers_info['virtualMachineRG']} -n DEFAULT " + \
                     f"-l {delta_workers_info['location']} " + \
                     f"-t azure_template/template.json " + \
                     f"-p azure_template/{resource_group_name}/{worker['name']}.json"

        res = subprocess.run(create_bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logging.info(f"run {create_bin} sucess.")
        pbar.update(1)
    pbar.close()
    
    for worker in delta_workers_info['virtualMachines']:
        get_IP_bin = f'''az network public-ip list -g {delta_workers_info['virtualMachineRG']} --query "[?name=='{worker["name"]}-ip']"'''
        res = subprocess.run(get_IP_bin, shell=True, capture_output=True)
        if res.returncode:
            logging.error(f"run {get_IP_bin} error! err msg: {res.stderr}")
            raise(res.stdout)
        else:
            logging.info(f"get IP of {worker['name']} success!")
            worker_ip = json.loads(res.stdout)[0]["ipAddress"]
            worker['IP'] = worker_ip

    with open(f'azure_template/resource_group_info/{resource_group_name}.json', 'r') as infile:
        exist_resource_group_info = json.load(infile)

    god_vnet_IP = exist_resource_group_info['virtualMachines'][0]['vnetIP']
    mount_samba_server(delta_workers_info, god_vnet_IP)
    install_docker(delta_workers_info)

    # questions = [
    #     inquirer.Text(
    #         'imageName', 
    #         message="What is the docker image name?",
    #         default="maro/ecr/cpu/latest"
    #     )
    # ]
    # image_name = inquirer.prompt(questions)['imageName']
    image_name = "docker_image"

    unpack_docker_images(delta_workers_info, image_name)

    # update resource group info json
    exist_resource_group_info["virtualMachines"].extend(delta_workers_info["virtualMachines"])

    with open(f'azure_template/resource_group_info/{resource_group_name}.json', 'w') as outfile:
        json.dump(exist_resource_group_info, outfile, indent=4)
    
    god_ip = exist_resource_group_info['virtualMachines'][0]['IP']
    res = subprocess.run(f"scp -o StrictHostKeyChecking=no azure_template/resource_group_info/{resource_group_name}.json {exist_resource_group_info['adminUsername']}@{god_ip}:~/resource_group_info.json", shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)


def inquirer_workers(resource_group_name):
    questions = [
        inquirer.Text(
            'workerSize',
            message="Which size is your worker machine?",
            default="Standard_D16s_v3"
        ),
        inquirer.Text(
            'workersNum',
            message="How many workers are you going to create?",
            default="5"
        ),
        inquirer.Text(
            'adminPublicKey',
            message="What is your public key? (default is ~/.ssh/id_rsa.pub)",
            default=open(f'/home/{getpass.getuser()}/.ssh/id_rsa.pub').read()
        )
    ]

    if not resource_group_name:
        questions += [
            inquirer.List(
                'virtualMachineRG', 
                message="Which resource group would you like to create workers?",
                choices=[resource_group_info[:-5] for resource_group_info in os.listdir(f"azure_template/resource_group_info")],
                carousel=True,
            ),
        ]

    worker_info = inquirer.prompt(questions)

    if resource_group_name:
        worker_info['virtualMachineRG'] = resource_group_name

    with open(f"azure_template/resource_group_info/{worker_info['virtualMachineRG']}.json", 'r') as infile:
        exist_resource_group_info = json.load(infile)
        exist_worker_num = len(exist_resource_group_info["virtualMachines"]) - 1

    worker_info = {
        "adminPublicKey": worker_info["adminPublicKey"],
        "adminUsername": exist_resource_group_info["adminUsername"],
        "virtualMachineRG": worker_info["virtualMachineRG"],
        "subscription": exist_resource_group_info["subscription"],
        "location": exist_resource_group_info["location"],
        "virtualMachines": [{
            "name": f"worker{exist_worker_num + i}",
            "size": worker_info["workerSize"]
        } for i in range(int(worker_info["workersNum"]))]
    }

    return worker_info

def mount_samba_server(delta_workers_info, god_vnet_IP):
    admin_username = delta_workers_info['adminUsername']
    for worker in delta_workers_info["virtualMachines"]:
        mount_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{worker['IP']} 'bash ADMIN_USERNAME={admin_username} SAMBA_IP={god_vnet_IP} /code_point/bin/mount_samba.sh'"
        res = subprocess.run(mount_bin, shell=True, capture_output=True)
        
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logging.info(f"run {mount_bin} success!")

def setup_samba_server():
    pass

def create_resource_group():
    resource_group_name, resource_group_info = create_god()
    init_god(resource_group_info)
    # create_workers(resource_group_name)

def generate_job_config():
    pass

# def generate_job_config():
#     PYTHONPATH = os.environ['PYTHONPATH']
#     default_config_path = os.path.join(PYTHONPATH, "examples/ecr/q_learning/distributed_mode/config.yml")

#     questions = [
#         inquirer.Text(
#             'configPath', 
#             message="Where is your meta config file?",
#             default=default_config_path
#         ),
#     ]

#     config_path = inquirer.prompt(questions)['configPath']

#     gen_job_config(config_path)

def generate_job_config():
    with open('azure_template/resource_group_info/maro_dist.json', 'r') as infile:
        resource_group_info = json.load(infile)
    init_god(resource_group_info)


def deploy_code():
    pass

if __name__=="__main__":
    create_god()