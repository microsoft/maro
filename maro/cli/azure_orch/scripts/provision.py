import getpass, os, subprocess
import logging
import inquirer
import json
import socket

import chalk
from tqdm import tqdm

from maro.cli.azure_orch.scripts.docker import install_docker, unpack_docker_images


logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('logger')

def create_god(god_info):
    logger.info(chalk.green("create god machine start!"))
    
    god_info = god_info
    resource_group_name = god_info['virtual_machine_resource_group']

    # if not os.path.exists('/maro'):
    #     os.mkdir('/maro')
    #     if not os.path.exists('/maro/dist')
    #         os.mkdir('/maro/dist')
    #         if not os.path.exists('/maro/dist/azure_template')
    #             os.mkdir('/maro/dist/azure_template')
    #             if not os.path.exists('/maro/dist/azure_template/resource_group_info')
    #                 os.mkdir('/maro/dist/azure_template/resource_group_info')

    if not os.path.exists('azure_template/resource_group_info'):
        os.mkdir('azure_template/resource_group_info')

    with open(f'azure_template/resource_group_info/{resource_group_name}.json', 'w') as outfile:  
        json.dump(god_info, outfile, indent=4)
    
    config_bin = f"python azure_template/vmconfig.py azure_template/resource_group_info/{resource_group_name}.json azure_template/{resource_group_name} azure_template/parameters.json"
    
    create_bin = f"./azure_template/deploy.sh -i {god_info['subscription']} " + \
                 f"-g {god_info['virtual_machine_resource_group']} -n DEFAULT " + \
                 f"-l {god_info['location']} " + \
                 f"-t azure_template/template.json " + \
                 f"-p azure_template/{resource_group_name}/god.json"

    # create god vm
    for bin in [config_bin, create_bin]:
        res = subprocess.run(bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info(f"run {bin} sucess.")
    
    # get god IP
    get_IP_bin = f"az network public-ip list -g {god_info['virtual_machine_resource_group']}"
    res = subprocess.run(get_IP_bin, shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)
    else:
        ip_info = json.loads(res.stdout)
        god_IP = ip_info[0]["ipAddress"]
    
    # get god vnet IP
    get_vnetIP_bin = "ip addr | grep 'state UP' -A2 | tail -n1 | awk '{print $2}' | cut -f1 -d '/'"
    res = subprocess.run(f"ssh -o StrictHostKeyChecking=no {god_info['admin_username']}@{god_IP} {get_vnetIP_bin}", shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)
    else:
        god_vnet_IP = str(res.stdout, 'utf-8')

    logger.info("Now you can remote login your god machine. ")
    logger.info(chalk.green(f"COPY AND RUN: ssh {god_info['admin_username']}@{god_IP}"))

    # update god IP and vnet IP    
    god_info['virtual_machines'][0]['IP'] = god_IP
    god_info['virtual_machines'][0]['vnet_IP'] = god_vnet_IP[:-1]

    # save god info
    with open(f'azure_template/resource_group_info/{resource_group_name}.json', 'w') as outfile:
        json.dump(god_info, outfile, indent=4)
    
    # upload god info to god vm
    res = subprocess.run(f"scp -o StrictHostKeyChecking=no azure_template/resource_group_info/{resource_group_name}.json {god_info['admin_username']}@{god_IP}:~/resource_group_info.json", shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)

def init_god(god_info, image_name="maro/ecr/cpu/latest"):
    admin_username = god_info["admin_username"]
    god_IP = god_info["virtual_machines"][0]["IP"]

    ssh_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{god_IP} "

    # gen public key
    gen_pubkey_bin = ssh_bin + '''"ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa <<< y"'''
    get_pubkey_bin = ssh_bin + "cat ~/.ssh/id_rsa.pub"
    for bin in [gen_pubkey_bin, get_pubkey_bin]:
        res = subprocess.run(bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            if "cat" in bin:
                god_public_key = str(res.stdout, 'utf-8') 
    
    # save god public key
    god_info['god_public_key'] = god_public_key
    resource_group_name = god_info['virtual_machine_resource_group']
    with open(f'azure_template/resource_group_info/{resource_group_name}.json', 'w') as outfile:
        json.dump(god_info, outfile, indent=4)

    # scp necessary files
    mkdir_bin = ssh_bin + "sudo mkdir /code_repo"
    chmod_bin = ssh_bin + "sudo chmod 777 /code_repo"
    scp_bash_bin = f"scp -o StrictHostKeyChecking=no -r bin {admin_username}@{god_IP}:/code_repo/"
    scp_redis_conf_bin = f"scp -o StrictHostKeyChecking=no -r redis_conf {admin_username}@{god_IP}:/code_repo/"
    scp_df_bin = f"scp -o StrictHostKeyChecking=no -r ../../../docker_files {admin_username}@{god_IP}:/code_repo/"
    scp_requirements_bin = f"scp -o StrictHostKeyChecking=no ../../../requirements.dev.txt {admin_username}@{god_IP}:/code_repo/"
    scp_prob_bin = f"scp -o StrictHostKeyChecking=no scripts/prob.py {admin_username}@{god_IP}:/code_repo/"
    for bin in [mkdir_bin, chmod_bin, scp_bash_bin, scp_redis_conf_bin, scp_df_bin, scp_requirements_bin, scp_prob_bin]:
        res = subprocess.run(bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info(f"run {bin} success!")

    # install docker
    install_docker_bin = ssh_bin + f"bash /code_repo/bin/install_docker.sh"

    # build docker images
    build_docker_images_bin = ssh_bin + f"sudo DOCKER_FILE=/code_repo/docker_files/cpu.dist.df DOCKER_FILE_DIR=/code_repo DOCKER_IMAGE_NAME={image_name} bash /code_repo/bin/build_image.sh"

    # save docker images to code point
    #TODO: support more docker files, change docker_image
    save_image_bin = ssh_bin + f"'sudo docker save {image_name} > /code_repo/docker_image'"

    for bin in [install_docker_bin, build_docker_images_bin, save_image_bin]:
        res = subprocess.run(bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info(f"run {bin} success!")
    
    # launch redis
    launch_redis_bin = ssh_bin + "sudo bash /code_repo/bin/launch_redis.sh"
    res = subprocess.run(launch_redis_bin, shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)
    else:
        logger.info(f"run {launch_redis_bin} success!")

    # setup samba file server
    setup_samba_bin = ssh_bin + "sudo bash /code_repo/bin/launch_samba.sh"
    res = subprocess.run(setup_samba_bin, shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)
    else:
        logger.info(f"setup samba file server on god success!")
    
    # set samba server password
    set_password_bin = ssh_bin + f"sudo smbpasswd -a {admin_username}"
    proc = subprocess.Popen(set_password_bin, shell=True, stdin=subprocess.PIPE)
    proc.communicate(input=b'''maro_dist\nmaro_dist''')

    return god_public_key

def create_nodes(delta_nodes_info=None):
    if not delta_nodes_info:
        delta_nodes_info = inquirer_delta_nodes()

    logger.info(chalk.green("create node machines start!"))

    delta_nodes_info_name = f"{delta_nodes_info['virtual_machine_resource_group']}.delta_info"
    resource_group_name = delta_nodes_info['virtual_machine_resource_group']
    
    if not os.path.exists('azure_template/delta_nodes_info'):
        os.mkdir('azure_template/delta_nodes_info')
    with open(f'azure_template/delta_nodes_info/{delta_nodes_info_name}.json', 'w') as outfile:  
        json.dump(delta_nodes_info, outfile, indent=4)

    config_bin = f"python azure_template/vmconfig.py azure_template/delta_nodes_info/{delta_nodes_info_name}.json azure_template/{resource_group_name} azure_template/parameters.json"
    
    res = subprocess.run(config_bin, shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)
    else:
        logging.info(f"run {config_bin} sucess.")

    pbar = tqdm(total=len(delta_nodes_info['virtual_machines']))

    # create nodes vm
    for node in delta_nodes_info['virtual_machines']:
        create_bin = f"./azure_template/deploy.sh -i {delta_nodes_info['subscription']} " + \
                     f"-g {delta_nodes_info['virtual_machine_resource_group']} -n DEFAULT " + \
                     f"-l {delta_nodes_info['location']} " + \
                     f"-t azure_template/template.json " + \
                     f"-p azure_template/{resource_group_name}/{node['name']}.json"

        res = subprocess.run(create_bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logging.info(f"run {create_bin} sucess.")
        pbar.update(1)
    pbar.close()
    
    # get nodes IP
    for node in delta_nodes_info['virtual_machines']:
        get_IP_bin = f'''az network public-ip list -g {delta_nodes_info['virtual_machine_resource_group']} --query "[?name=='{node["name"]}-ip']"'''
        res = subprocess.run(get_IP_bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stdout)
        else:
            logging.info(f"get IP of {node['name']} success!")
            node_ip = json.loads(res.stdout)[0]["ipAddress"]
            node['IP'] = node_ip

    with open(f'azure_template/resource_group_info/{resource_group_name}.json', 'r') as infile:
        exist_resource_group_info = json.load(infile)

    # provision infrastructures
    god_vnet_IP = exist_resource_group_info['virtual_machines'][0]['vnet_IP']
    image_name = "docker_image"     # docker image name simply set to docker_image
    mount_samba_server(delta_nodes_info, god_vnet_IP)
    install_docker(delta_nodes_info)
    deploy_god_pubkey(delta_nodes_info)
    unpack_docker_images(delta_nodes_info, image_name)

    # update resource group info json and save
    exist_resource_group_info["virtual_machines"].extend(delta_nodes_info["virtual_machines"])
    with open(f'azure_template/resource_group_info/{resource_group_name}.json', 'w') as outfile:
        json.dump(exist_resource_group_info, outfile, indent=4)
    
    # upload resource group info to god vm
    god_IP = exist_resource_group_info['virtual_machines'][0]['IP']
    res = subprocess.run(f"scp -o StrictHostKeyChecking=no azure_template/resource_group_info/{resource_group_name}.json {exist_resource_group_info['admin_username']}@{god_IP}:~/resource_group_info.json", shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)

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
            'admin_username', 
            message="Who is the admin user on resource group?",
        ),
        inquirer.Text(
            'admin_public_key',
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
            message="In which location you will set up the resource group?",
            default="southeastasia"
        ),
        inquirer.Text(
            'virtual_machine_resource_group',
            message="What name is your group?",
            default="maro_dist"
        ),
        inquirer.Text(
            'god_size',
            message="Which size is your god machine?",
            default="Standard_D16s_v3"
        ),
        inquirer.Text(
            'node_size',
            message="Which size is your node machine?",
            default="Standard_D16s_v3"
        ),
        inquirer.Text(
            'nodes_num',
            message="How many nodes are you going to create?",
            default="5"
        ),
    ]

    resource_group_info = inquirer.prompt(questions)

    god_info = {
        "admin_public_key": resource_group_info["admin_public_key"],
        "admin_username": resource_group_info["admin_username"],
        "virtual_machine_resource_group": resource_group_info["virtual_machine_resource_group"],
        "subscription": resource_group_info["subscription"].split(": ")[1],
        "location": resource_group_info["location"],
        "virtual_machines": [{
            "name": "god",
            "size": resource_group_info["god_size"]
        }]
    }

    nodes_info = {
        "admin_public_key": resource_group_info["admin_public_key"],
        "admin_username": resource_group_info["admin_username"],
        "virtual_machine_resource_group": resource_group_info["virtual_machine_resource_group"],
        "subscription": resource_group_info["subscription"].split(": ")[1],
        "location": resource_group_info["location"],
        "virtual_machines": [{
            "name": f"node{i}",
            "size": resource_group_info["node_size"]
        } for i in range(int(resource_group_info["nodes_num"]))]
    }

    return god_info, nodes_info

def inquirer_delta_nodes():
    questions = [
        inquirer.Text(
            'node_size',
            message="Which size is your node machine?",
            default="Standard_D16s_v3"
        ),
        inquirer.Text(
            'nodes_num',
            message="How many nodes are you going to create?",
            default="5"
        ),
        inquirer.List(
            'virtual_machine_resource_group', 
            message="Which resource group would you like to create nodes?",
            choices=[resource_group_info[:-5] for resource_group_info in os.listdir(f"azure_template/resource_group_info")],
            carousel=True,
        ),
    ]

    delta_nodes_info = inquirer.prompt(questions)

    with open(f"azure_template/resource_group_info/{delta_nodes_info['virtual_machine_resource_group']}.json", 'r') as infile:
        exist_resource_group_info = json.load(infile)
        exist_node_num = len(exist_resource_group_info["virtual_machines"]) - 1

    delta_nodes_info = {
        "admin_public_key": exist_resource_group_info["admin_public_key"],
        "god_public_key": exist_resource_group_info["god_public_key"],
        "admin_username": exist_resource_group_info["admin_username"],
        "virtual_machine_resource_group": exist_resource_group_info["virtual_machine_resource_group"],
        "subscription": exist_resource_group_info["subscription"],
        "location": exist_resource_group_info["location"],
        "virtual_machines": [{
            "name": f"node{exist_node_num + i}",
            "size": delta_nodes_info["node_size"]
        } for i in range(int(delta_nodes_info["nodes_num"]))]
    }

    return delta_nodes_info

def mount_samba_server(delta_nodes_info, god_vnet_IP):
    admin_username = delta_nodes_info['admin_username']
    mkdir_bin = "sudo mkdir /code_repo"
    chmod_bin = "sudo chmod 777 /code_repo"
    mount_bin = f"sudo mount -t cifs -o username={admin_username},password=maro_dist //{god_vnet_IP}/sambashare /code_repo"
    append_bin = f'''"echo '//{god_vnet_IP}/sambashare  /code_repo cifs  username={admin_username},password=maro_dist  0  0' | sudo tee -a /etc/fstab"'''
    
    for node in delta_nodes_info["virtual_machines"]:
        ssh_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{node['IP']} "
        ssh_mkdir_bin = ssh_bin + mkdir_bin
        ssh_mount_bin = ssh_bin + mount_bin
        ssh_chmod_bin = ssh_bin + chmod_bin
        ssh_append_bin = ssh_bin + append_bin

        for bin in [ssh_mkdir_bin, ssh_mount_bin, ssh_chmod_bin, ssh_append_bin]:
            res = subprocess.run(bin, shell=True, capture_output=True)
        
            if res.returncode:
                raise Exception(res.stderr)
            else:
                logging.info(f"run {bin} success!")

def deploy_god_pubkey(delta_nodes_info):
    admin_username = delta_nodes_info['admin_username']
    god_public_key = delta_nodes_info['god_public_key']
    for node in delta_nodes_info["virtual_machines"]:
        deploy_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{node['IP']} 'echo {god_public_key} >> ~/.ssh/authorized_keys'"
        res = subprocess.run(deploy_bin, shell=True, capture_output=True)
        
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logging.info(f"run {deploy_bin} success!")

def create_resource_group():
    god_info, nodes_info = inquirer_resource_group()
    create_god(god_info)
    god_public_key = init_god(god_info)
    nodes_info['god_public_key'] = god_public_key
    create_nodes(nodes_info)

def increase_resource_group():
    create_nodes()

def decrease_resource_group():
    pass

def stop_nodes():
    questions = [
        inquirer.List(
            'virtual_machine_resource_group', 
            message="Which resource group would you like to stop all nodes?",
            choices=[resource_group_info[:-5] for resource_group_info in os.listdir(f"azure_template/resource_group_info")],
            carousel=True,
        ),
    ]
    resource_group_name = inquirer.prompt(questions)['virtual_machine_resource_group']

    with open(f'azure_template/resource_group_info/{resource_group_name}') as infile:
        resource_group_info = json.load(infile)

    logger.info(f"Resource Group: {resource_group_info['virtual_machine_resource_group']}")

    logger.info("Machines to stop:")
    logger.info(' '.join([node["name"] for node in resource_group_info["virtual_machines"]]))

    questions = [
        inquirer.Text(
            'firstAffirm', 
            message="Are you sure to stop up all of them? (y/n)",
                    ),
        inquirer.Text(
            'secondAffirm',
            message="Really? (y/n)",        
            )
    ]
    answers = inquirer.prompt(questions)

    if answers["firstAffirm"] == "y" and answers["secondAffirm"] == "y":
        for node in resource_group_info["virtual_machines"]:
            stop_bin = f"(az vm deallocate -g {resource_group_name} -n {node['name']} || echo 'stop {node['name']} error') &"
        res = subprocess.run(stop_bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info("stop cluster success!")

def start_nodes():
    questions = [
        inquirer.List(
            'virtual_machine_resource_group', 
            message="Which resource group would you like to start all nodes?",
            choices=[resource_group_info[:-5] for resource_group_info in os.listdir(f"azure_template/resource_group_info")],
            carousel=True,
        ),
    ]
    resource_group_name = inquirer.prompt(questions)['virtual_machine_resource_group']

    with open(f'azure_template/resource_group_info/{resource_group_name}') as infile:
        resource_group_info = json.load(infile)

    logger.info(f"Resource Group: {resource_group_info['virtual_machine_resource_group']}")

    logger.info("Machines to start:")
    logger.info(' '.join([node["name"] for node in resource_group_info["virtual_machines"]]))

    questions = [
        inquirer.Text(
            'firstAffirm', 
            message="Are you sure to start up all of them? (y/n)",
                    ),
        inquirer.Text(
            'secondAffirm',
            message="Really? (y/n)",        
            )
    ]
    answers = inquirer.prompt(questions)

    if answers["firstAffirm"] == "y" and answers["secondAffirm"] == "y":
        for node in resource_group_info["virtual_machines"]:
            start_bin = f"(az vm start -g {resource_group_name} -n {node['name']} || echo 'start {node['name']} error') &"
        res = subprocess.run(start_bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info("start cluster success!")