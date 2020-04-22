import getpass, os, subprocess
import logging
import inquirer
import json
import socket

import chalk
from tqdm import tqdm


logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('logger')

def create_god():
    resouce_group_info = inquirer_resource_group()
    resouce_group_name = resouce_group_info['virtualMachineRG']

    if not os.path.exists('azure_template/resouce_group_info'):
        os.mkdir('azure_template/resouce_group_info')

    with open(f'azure_template/resouce_group_info/{resouce_group_name}.json', 'w') as outfile:  
        json.dump(resouce_group_info, outfile, indent=4)
    
    config_bin = f"python azure_template/vmconfig.py azure_template/resouce_group_info/{resouce_group_name}.json azure_template/{resouce_group_name} azure_template/parameters.json"
    
    create_bin = f"./azure_template/deploy.sh -i {resouce_group_info['subscription']} " + \
                 f"-g {resouce_group_info['virtualMachineRG']} -n DEFAULT " + \
                 f"-l {resouce_group_info['location']} " + \
                 f"-t azure_template/template.json " + \
                 f"-p azure_template/{resouce_group_name}/god.json"

    get_IP_bin = f"az network public-ip list -g {resouce_group_info['virtualMachineRG']}"

    for bin in [config_bin, create_bin, get_IP_bin]:
        res = subprocess.run(bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            if bin[0] == 'a':
                ip_info = json.loads(res.stdout)
                god_ip = ip_info[0]["ipAddress"]
                resouce_group_info['virtualMachines'][0]['IP'] = god_ip
                logger.info("Now you can remote login your god machine. ")
                logger.info(chalk.red(f"COPY AND RUN: ssh {resouce_group_info['adminUsername']}@{god_ip}"))
                if res.returncode:
                    raise Exception(res.stderr)
            else:
                logger.info(f"run {bin} sucess.")

    with open(f'azure_template/resouce_group_info/{resouce_group_name}.json', 'w') as outfile:
        json.dump(resouce_group_info, outfile, indent=4)

    return resouce_group_name, resouce_group_info

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

    resouce_group_info = inquirer.prompt(questions)

    resouce_group_info = {
        "adminPublicKey": resouce_group_info["adminPublicKey"],
        "adminUsername": resouce_group_info["adminUsername"],
        "virtualMachineRG": resouce_group_info["virtualMachineRG"],
        "subscription": resouce_group_info["subscription"].split(": ")[1],
        "location": resouce_group_info["location"],
        "virtualMachines": [{
            "name": "god",
            "size": resouce_group_info["godSize"]
        }]
    }

    return resouce_group_info


def init_god(resouce_group_info):
    admin_username = resouce_group_info["adminUsername"]
    god_ip = resouce_group_info["virtualMachines"][0]["IP"]
    ssh_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{god_ip} "

    # gen pub key
    gen_pubkey_bin = ssh_bin + "ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa <<< y"

    # setup samba file server


    # scp necessary files
    scp_bash_bin = f"scp -o StrictHostKeyChecking=no -r bin {admin_username}@{god_ip}:/code_point/"
    scp_df_bin = f"scp -o StrictHostKeyChecking=no -r ../../docker_files {admin_username}@{god_ip}:/code_point/"
    scp_requirements_bin = f"scp -o StrictHostKeyChecking=no ../../requirements.dev.txt {admin_username}@{god_ip}:/code_point/"

    # install docker
    install_docker_bin = ssh_bin + f"bash /code_point/bin/install_docker.sh"

    # build docker images
    build_docker_images_bin = ssh_bin + f"sudo DOCKER_FILE=/codepoint/docker_files/cpu.dist.df DOCKER_FILE_DIR=/codepoint DOCKER_IMAGE_NAME={image_name} bash /codepoint/bin/build_image.sh"

    # pack to code point

def create_workers(resource_group_name=None):
    delta_workers_info = inquirer_workers(resource_group_name)

    delta_workers_info_name = f"{delta_workers_info['virtualMachineRG']}.delta_info"
    resouce_group_name = delta_workers_info['virtualMachineRG']
    
    with open(f'azure_template/{delta_workers_info_name}.json', 'w') as outfile:  
        json.dump(delta_workers_info, outfile, indent=4)

    config_bin = f"python azure_template/vmconfig.py azure_template/{delta_workers_info_name}.json azure_template/{resouce_group_name} azure_template/parameters.json"
    
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
                     f"-p azure_template/{resouce_group_name}/{worker['name']}.json"

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

    
    mount_samba_server(delta_workers_info)
    install_docker(delta_workers_info)

    questions = [
        inquirer.Text(
            'imageName', 
            message="What is the docker image name?",
            default="maro/ecr/cpu/latest"
        )
    ]
    image_name = inquirer.prompt(questions)['imageName']

    build_cpu_docker_images(delta_workers_info, image_name)
    prob_resources(delta_workers_info, image_name)


    with open(f'azure_template/resouce_group_info/{resouce_group_name}.json', 'r') as infile:
        exist_resource_group_info = json.load(infile)

    # update resource group info json
    exist_resource_group_info["virtualMachines"].extend(delta_workers_info["virtualMachines"])

    with open(f'azure_template/resouce_group_info/{resouce_group_name}.json', 'w') as outfile:
        json.dump(exist_resource_group_info, outfile, indent=4)


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
                choices=os.listdir(f"azure_template/resouce_group_info"),
                carousel=True,
            ),
        ]

    worker_info = inquirer.prompt(questions)

    if resource_group_name:
        worker_info['virtualMachineRG'] = resource_group_name

    with open(f"azure_template/resouce_group_info/{worker_info['virtualMachineRG']}.json", 'r') as infile:
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

def mount_samba_server():
    pass

def setup_samba_server():
    pass

def create_resource_group():
    resouce_group_name, resouce_group_info = create_god()
    init_god(resouce_group_info)
    create_workers(resouce_group_name)

def generate_job_config():
    pass


