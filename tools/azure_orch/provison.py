import getpass, os, subprocess
import logging
from dirsync import sync
import inquirer
import json
import socket

from tqdm import tqdm

from docker import install_docker, build_cpu_docker_images
from gen_job_config import gen_job_config

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def config_summary(nodes_config):
    pass

def create_god():
    god_info = inquirer_god()
    god_info_name = f"god_info_{god_info['virtualMachineRG']}"
    with open(f'azure_template/{god_info_name}.json', 'w') as outfile:  
        json.dump(god_info, outfile, indent=4)
    
    config_bin = f"python azure_template/vmconfig.py azure_template/{god_info_name}.json azure_template/config_{god_info['virtualMachineRG']} azure_template/parameters.json"
    
    create_bin = f"./azure_template/deploy.sh -i {god_info['subscription']} " + \
                 f"-g {god_info['virtualMachineRG']} -n DEFAULT " + \
                 f"-l {god_info['location']} " + \
                 f"-t azure_template/template.json " + \
                 f"-p azure_template/config_{god_info['virtualMachineRG']}/god.json"

    get_IP_bin = f"az network public-ip list -g {god_info['virtualMachineRG']}"

    for bin in [config_bin, create_bin, get_IP_bin]:
        res = subprocess.run(bin, shell=True, capture_output=True)
        if res.returncode:
            logging.error(f"run {bin} error! err msg: {res.stderr}")
            raise("!!!")
        else:
            if bin[0] == 'a':
                with open(f"azure_template/config_{god_info['virtualMachineRG']}/god.json", 'r') as infile:  
                    god_deploy_config = json.load(infile)
                storage_account_name = god_deploy_config["parameters"]["diagnosticsStorageAccountName"]['value']

                with open(f'azure_template/{god_info_name}.json', 'r') as infile:
                    god_info = json.load(infile)
                god_info['storageAccountName'] = storage_account_name

                with open(f'azure_template/{god_info_name}.json', 'w') as outfile: 
                    json.dump(god_info, outfile, indent=4)

                ip_info = json.loads(res.stdout)
                god_ip = ip_info[0]["ipAddress"]
                logging.info("Now you can remote login your agent machine. ")
                logging.info(f"COPY AND RUN: ssh {god_info['adminUsername']}@{god_ip}")
                god_info['virtualMachines'][0]['IP'] = god_ip
                res = subprocess.run(f"scp -o StrictHostKeyChecking=no azure_template/{god_info_name}.json {god_info['adminUsername']}@{god_ip}:~/clusterInfo.json", 
                                    shell=True, capture_output=True)
                if res.returncode:
                    logging.error(f"scp error! err msg: {res.stderr}")
                    raise("!!!")
            else:
                logging.info(f"run {bin} sucess.")

    create_file_share(god_info)
    mount_AFS(god_info, god_deploy_config["parameters"]["diagnosticsStorageAccountName"]['value'])
    
def create_file_share(god_info):
    with open(f"azure_template/config_{god_info['virtualMachineRG']}/god.json", 'r') as infile:  
        god_info = json.load(infile)
    
    res = subprocess.run(f"storageAccountName={god_info['parameters']['diagnosticsStorageAccountName']['value']} \
        resourceGroupName={god_info['parameters']['virtualMachineRG']['value']} \
            shareName=sharefile \
                bash ./bin/create_file_share.sh",
                shell=True, capture_output=True)

    if res.returncode:
        logging.error(f"share file create error! err msg: {res.stderr}")
        raise("!!!")
    else:
        logging.info("share file create success!")


def mount_AFS(delta_cluster_info, storage_account_name):
    admin_username = delta_cluster_info['adminUsername']

    for worker in delta_cluster_info["virtualMachines"]:
        share_name = "sharefile"
        share_file_location = f"//{storage_account_name}.file.core.windows.net/{share_name} "
        mount_dir = "/codepoint/ "
        password = subprocess.run(f'''az storage account keys list \
                                    --resource-group {delta_cluster_info["virtualMachineRG"]} \
                                    --account-name {storage_account_name} \
                                    --query "[0].value" | tr -d '"' ''', 
                                    shell=True, capture_output=True, encoding='ascii').stdout
        password = password[:-1]

        auth = f"vers=3.0,username={storage_account_name},password={password},dir_mode=0777,file_mode=0777,sec=ntlmssp"
        
        mount_bin = "sudo mkdir -p /codepoint; " + "sudo mount -t cifs " + share_file_location + mount_dir + "-o " + auth
        ssh_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{worker['IP']} '{mount_bin}'"

        res = subprocess.run(ssh_bin, shell=True, capture_output=True)
        
        if res.returncode:
            logging.error(f"{worker['name']} mount AFS failed! msg err: {res.stderr}")
            raise("!!!")
        else:
            logging.info(f"{worker['name']} mount AFS success!")


def create_cluster():
    delta_cluster_info = inquirer_cluster()

    with open(f'/home/{getpass.getuser()}/clusterInfo.json', 'r') as infile:
        exist_cluster_info = json.load(infile)

    cluster_info_name = "delta_cluster_info"
    
    with open(f'azure_template/{cluster_info_name}.json', 'w') as outfile:  
        json.dump(delta_cluster_info, outfile, indent=4)

    config_bin = f"python azure_template/vmconfig.py azure_template/{cluster_info_name}.json azure_template/config azure_template/parameters.json"
    
    res = subprocess.run(config_bin, shell=True, capture_output=True)
    if res.returncode:
        logging.error(f"run {config_bin} error. error msg: {res.stderr}")
        raise("!!!")
    else:
        logging.info(f"run {config_bin} sucess.")

    pbar = tqdm(total=len(delta_cluster_info['virtualMachines']))

    for worker in os.listdir(f"azure_template/config"):
        create_bin = f"./azure_template/deploy.sh -i {delta_cluster_info['subscription']} " + \
                     f"-g {delta_cluster_info['virtualMachineRG']} -n DEFAULT " + \
                     f"-l {delta_cluster_info['location']} " + \
                     f"-t azure_template/template.json " + \
                     f"-p azure_template/config/{worker}"

        res = subprocess.run(create_bin, shell=True, capture_output=True)
        if res.returncode:
            logging.error(f"run {create_bin} error! err msg: {res.stderr}")
            raise("!!!")
        else:
            logging.info(f"run {create_bin} sucess.")
        pbar.update(1)
    pbar.close()
    
    
    for worker in delta_cluster_info['virtualMachines']:
        get_IP_bin = f'''az network public-ip list -g {delta_cluster_info['virtualMachineRG']} --query "[?name=='{worker["name"]}-ip']"'''
        res = subprocess.run(get_IP_bin, shell=True, capture_output=True)
        if res.returncode:
            logging.error(f"run {get_IP_bin} error! err msg: {res.stderr}")
            raise(res.stdout)
        else:
            logging.info(f"get IP of {worker['name']} success!")
            worker_ip = json.loads(res.stdout)[0]["ipAddress"]
            worker['IP'] = worker_ip

    
    mount_AFS(delta_cluster_info, exist_cluster_info['storageAccountName'])
    install_docker(delta_cluster_info)

    questions = [
        inquirer.Text(
            'imageName', 
            message="What is the docker image name?",
            default="maro/ecr/cpu/latest"
        )
    ]
    image_name = inquirer.prompt(questions)['imageName']

    build_cpu_docker_images(delta_cluster_info, image_name)
    prob_resources(delta_cluster_info, image_name)

    # update cluster info json
    exist_cluster_info["virtualMachines"].extend(delta_cluster_info["virtualMachines"])

    with open(f'/home/{getpass.getuser()}/clusterInfo.json', 'w') as outfile:
        json.dump(exist_cluster_info, outfile, indent=4)


def init_god():
    #sync code to codepoint
    src = os.environ['PYTHONPATH']
    dest = "/codepoint/"
    sync(src, "/codepoint/", 'sync', purge=True)

    #initialize docker
    install_bin = "bash /codepoint/tools/azure_orch/bin/install_docker.sh"
    res = subprocess.run(install_bin, shell=True, capture_output=True)

    if res.returncode:
        logging.error(f"run {install_bin} failed! err msg: {res.stderr}")
        raise("!!!")
    else:
        logging.info(f"run {install_bin} success!")

    #launch redis-server
    launch_bin = "bash /codepoint/tools/azure_orch/bin/launch_redis.sh"
    res = subprocess.run(launch_bin, shell=True, capture_output=True)

    if res.returncode:
        logging.error(f"run {launch_bin} failed! err msg: {res.stderr}")
        raise("!!!")
    else:
        logging.info(f"run {launch_bin} success!")
    
    #install az
    install_bin = "bash /codepoint/tools/azure_orch/bin/install_az.sh"
    res = subprocess.run(install_bin, shell=True, capture_output=True)

    if res.returncode:
        logging.error(f"run {install_bin} failed! err msg: {res.stderr}")
        raise("!!!")
    else:
        logging.info(f"run {install_bin} success!")
    
    #gen sshkey
    sshkey_bin = "ssh-keygen"
    res = subprocess.run(sshkey_bin, shell=True, capture_output=True)

    if res.returncode:
        logging.error(f"run {sshkey_bin} failed! err msg: {res.stderr}")
        raise("!!!")
    else:
        logging.info(f"run {sshkey_bin} success!")
    

def inquirer_cluster():
    questions = [
        inquirer.Text(
            'workerSize',
            message="Which size is your worker machine?",
            default="Standard_D16s_v3"
        ),
        inquirer.Text(
            'workersNum',
            message="How many workers are you going to create",
            default="5"
        ),
        inquirer.Text(
            'adminPublicKey',
            message="What is your public key? (default is ~/.ssh/id_rsa.pub)",
            default=open(f'/home/{getpass.getuser()}/.ssh/id_rsa.pub').read()
        )
    ]
    cluster_info = inquirer.prompt(questions)

    with open(f'/home/{getpass.getuser()}/clusterInfo.json', 'r') as infile:
        exist_cluster_info = json.load(infile)
        exist_cluster_num = len(exist_cluster_info["virtualMachines"]) - 1
    

    cluster_info = {
        "adminPublicKey": cluster_info["adminPublicKey"],
        "adminUsername": exist_cluster_info["adminUsername"],
        "virtualMachineRG": exist_cluster_info["virtualMachineRG"],
        "subscription": exist_cluster_info["subscription"],
        "location": exist_cluster_info["location"],
        "virtualMachines": [{
            "name": f"worker{exist_cluster_num + i}",
            "size": cluster_info["workerSize"]
        } for i in range(int(cluster_info["workersNum"]))]
    }

    return cluster_info

def inquirer_god():
    questions = [
        inquirer.Text(
            'adminUsername', 
            message="Who is the admin user on god?",
        ),
        inquirer.Text(
            'adminPublicKey',
            message="What is your public key? (default is ~/.ssh/id_rsa.pub)",
            # default=open(f'/home/{getpass.getuser()}/.ssh/id_rsa.pub').read()
            default=open(f'/home/v-tiansu/.ssh/id_rsa.pub').read()
        ),
        inquirer.Text(
            'subscription',
            message="Set your subscription: (default is for Research-ARD Incubation-MSRA)",
            default="e225d10c-2fce-4a2c-8ab8-fbf0cf28b99e",
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

    god_info = inquirer.prompt(questions)

    god_info = {
        "adminPublicKey": god_info["adminPublicKey"],
        "adminUsername": god_info["adminUsername"],
        "virtualMachineRG": god_info["virtualMachineRG"],
        "subscription": god_info["subscription"],
        "location": god_info["location"],
        "virtualMachines": [{
            "name": "god",
            "size": god_info["godSize"]
        }]
    }

    return god_info


def prob_resources(delta_cluster_info, image_name):
    redis_address = socket.gethostbyname(socket.gethostname())
    redis_port = 6379
    admin_username = delta_cluster_info["adminUsername"]
    for worker in delta_cluster_info["virtualMachines"]:
        prob_bin = f"docker run --name prob -d -it -v /codepoint:/maro_dist {image_name} REDIS_ADDRESS={redis_address} REDIS_PORT={redis_port} python3 tools/azure_orch/prob.py"
        ssh_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{worker['IP']} '{prob_bin}'"

        res = subprocess.run(ssh_bin, shell=True)
        
        if res.returncode:
            logging.error(f"start prob on {worker['name']} failed! err msg: {res.stderr}")
            raise("!!!")
        else:
            logging.info(f"start prob on {worker['name']} success!")

def start_cluster(cluster_info):
    logging.info(f"Resource Group: {cluster_info['virtualMachineRG']}")

    logging.info("Machines to start:")
    logging.info(' '.join([worker["name"] for worker in cluster_info["virtualMachines"]]))

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

    start_bin = ""
    virtualMachineRG = cluster_info["virtualMachineRG"]
    if answers["firstAffirm"] == "y" and answers["secondAffirm"] == "y":
        for worker in cluster_info["virtualMachines"]:
            start_bin = f"(az vm start -g {virtualMachineRG} -n {worker['name']} || echo 'start {worker['name']} error') &"
        res = subprocess.run(start_bin, shell=True, capture_output=True)

        if res.returncode:
            logging.error("start cluster failed! err msg: {res.stderr}")
            raise("!!!")
        else:
            logging.info("start cluster success!")

def stop_cluster(cluster_info):
    # with open(f'~/clusterInfo.json', 'r') as infile:
    #     exist_cluster_info = json.load(infile)
    
    logging.info(f"Resource Group: {cluster_info['virtualMachineRG']}")

    logging.info("Machines to start:")
    logging.info(' '.join([worker["name"] for worker in cluster_info["virtualMachines"]]))

    questions = [
        inquirer.Text(
            'firstAffirm', 
            message="Are you sure to stop all of them? (y/n)",
        ),
        inquirer.Text(
            'secondAffirm',
            message="Really? (y/n)",
        )
    ]

    answers = inquirer.prompt(questions)

    start_bin = ""
    virtualMachineRG = cluster_info["virtualMachineRG"]
    if answers["firstAffirm"] == "y" and answers["secondAffirm"] == "y":
        for worker in cluster_info["virtualMachines"]:
            start_bin = f"(az vm deallocate -g {virtualMachineRG} -n {worker['name']} || echo 'stop {worker['name']} error') &"
        res = subprocess.run(start_bin, shell=True, capture_output=True)

        if res.returncode:
            logging.error(f"stop cluster failed! err msg: {res.stderr}")
            raise("!!!")
        else:
            logging.info(f"stop cluster success!")

def generate_job_config():
    questions = [
        inquirer.Text(
            'configPath', 
            message="Where is your config file?",
            default="/codepoint/examples/ecr/q_learning/distributed_mode/config.yml"
        ),
    ]

    config_path = inquirer.prompt(questions)['configPath']

    gen_job_config(config_path)


# unit test
if __name__ == "__main__":
    # create_god()
    generate_job_config()