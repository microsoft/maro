import getpass, os, subprocess
from utils import output_level, output
import inquirer
import json

def config_summary(nodes_config):
    pass

# def create_cluster(nodes_config):
#     config_summary(nodes_config)
#     output(output_level.YELLOW, "this is a test!")
#     # for node in nodes_config:
#     #     if node.type == "redis_agent":

#     #     else:

def create_cluster():
    delta_cluster_info = inquirer_cluster()
    with open(f'~/clusterInfo.json', 'r') as infile:
        exist_cluster_info = json.load(infile)
    
    # new cluster info json
    exist_cluster_info["virtualMachines"].extend(delta_cluster_info["virtualMachines"])

    cluster_info_name = "delta_cluster_info"
    
    with open(f'azure_template/{cluster_info_name}.json', 'w') as outfile:  
        json.dump(delta_cluster_info, outfile, indent=4)

    config_bin = f"python azure_template/vmconfig.py azure_template/{cluster_info_name}.json azure_template/config azure_template/parameters.json"
    
    res = subprocess.run(config_bin, shell=True, capture_output=True)
    if res.returncode:
        output(output_level.RED, f"run {config_bin} error.")
        raise("!!!")
    else:
         output(output_level.WHITE, f"run {config_bin} sucess.")

    for worker in os.listdir(f"azure_template/config"):
        create_bin = f"./azure_template/deploy.sh -i {delta_cluster_info['subscription']} " + \
                     f"-g {delta_cluster_info['virtualMachineRG']} -n DEFAULT " + \
                     f"-l {delta_cluster_info['location']} " + \
                     f"-t azure_template/template.json " + \
                     f"-p azure_template/config/{worker}"

        res = subprocess.run(create_bin, shell=True, capture_output=True)
        if res.returncode:
            output(output_level.RED, f"run {create_bin} error.")
            raise("!!!")
        else:
            output(output_level.WHITE, f"run {create_bin} sucess.")
    
    start_cluster(delta_cluster_info)
    mount_AFS(delta_cluster_info)

    with open(f'~/clusterInfo.json', 'w') as outfile:
        json.dump(exist_cluster_info, outfile, indent=4)

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
            output(output_level.RED, f"run {bin} error.")
            raise("!!!")
        else:
            if bin[0] == 'a':
                ip_info = json.loads(res.stdout)
                god_ip = ip_info[0]["ipAddress"]
                output(output_level.WHITE, "Now you can remote login your agent machine.", end="")
                output(output_level.RED, "COPY AND RUN:", end="")
                output(output_level.WHITE, f"ssh {god_info['adminUsername']}@{god_ip}")
                res = subprocess.run(f"scp -o StrictHostKeyChecking=no azure_template/CreationMemorandum.md {god_info['adminUsername']}@{god_ip}:~; " + \
                                f"scp -o StrictHostKeyChecking=no azure_template/{god_info_name}.json {god_info['adminUsername']}@{god_ip}:~/clusterInfo.json", 
                                shell=True, capture_output=True)
                if res.returncode:
                    output(output_level.RED, f"scp error.")
                    raise("!!!")
            else:
                output(output_level.WHITE, f"run {bin} sucess.")
    
    create_file_share(god_info)
    
def create_file_share(god_info):
    with open(f"azure_template/config_{god_info['virtualMachineRG']}/{god_info['virtualMachines'][0]['name']}.json", 'w') as infile:  
        god_info = json.load(infile)
    
    res = subprocess.run(f"storageAccountName={god_info['parameters']['diagnosticsStorageAccountName']} \
        resourceGroupName={god_info['virtualMachineRG']} \
            shareName={god_info['virtualMachineRG']} \
                ./bin/create_file_share.sh")

    if res.returncode:
        output(output_level.RED, "share file create failed!")
        raise("!!!")
    else:
        output(output_level.WHITE, "share file create success!")

def mount_AFS(delta_cluster_info):
    with open(f"azure_template/config_{delta_cluster_info['virtualMachineRG']}/god.json", 'w') as infile:  
        god_info = json.load(infile)
    storage_account_name = god_info["parameters"]["diagnosticsStorageAccountName"]

    for worker in delta_cluster_info["virtualMachines"]:
        share_name = delta_cluster_info["virtualMachineRG"]
        share_file_location = f"//{storage_account_name}.file.core.windows.net/{share_name} "
        mount_dir = "/docker_images/ "
        password = subprocess.run(f'''az storage account keys list \
                                    --resource-group {delta_cluster_info["virtualMachineRG"]} \
                                    --account-name {storage_account_name} \
                                    --query "[0].value" | tr -d '"' ''', 
                                    shell=True, capture_output=True).stdout

        auth = f"vers=3.0,username={share_name},password={password},dir_mode=0777,file_mode=0777,sec=ntlmssp"
        
        mount_bin = "sudo mkdir -p /docker_images; " + "sudo mount -t cifs " + share_file_location + mount_dir + "-o " + auth
        ssh_bin = f"ssh -o StrictHostKeyChecking=no {worker['name']} '{mount_bin}'"

        res = subprocess.run(ssh_bin, shell=True, capture_output=True)
        
        if res.returncode:
            output(output_level.RED, f"{worker['name']} mount AFS failed!")
            raise("!!!")
        else:
            output(output_level.WHITE, f"{worker['name']} mount AFS success!")


def inquirer_cluster():
    questions = [
        inquirer.Text(
            'workerSize',
            message="Which size is your worker machine?",
            default="Standard_D16s_v3"
        ),
        inquirer.Text(
            'wokersNum',
            message="How many workers are you going to create",
            default="5"
        )
    ]
    cluster_info = inquirer.prompt(questions)

    with open(f'~/clusterInfo.json', 'r') as infile:
        exist_cluster_info = json.load(infile)
        exist_cluster_num = len(exist_cluster_info["virtualMachines"]) - 1

    cluster_info = {
        "adminPublicKey": exist_cluster_info["adminPublicKey"],
        "adminUsername": exist_cluster_info["adminUsername"],
        "virtualMachineRG": exist_cluster_info["virtualMachineRG"],
        "subscription": exist_cluster_info["subscription"],
        "location": exist_cluster_info["location"],
        "virtualMachines": [{
            "name": f"worker_{exist_cluster_num + i}",
            "size": cluster_info["workerSize"]
        } for i in range(cluster_info["workersNum"])]
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
            default="UNDEFINED"
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


def start_cluster(group_name):
    output(output_level.YELLOW, "Resource Group:", end="")
    output(output_level.WHITE, delta_cluster_info["virtualMachineRG"])

    output(output_level.YELLOW, "Machines to start:")
    for worker in delta_cluster_info["virtualMachines"]:
        output(output_level.WHITE, worker["name"])

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
    virtualMachineRG = delta_cluster_info["virtualMachineRG"]
    if answers["firstAffirm"] == "y" and answers["secondAffirm"] == "y":
        for worker in delta_cluster_info["virtualMachines"]:
            start_bin = f"(az vm start -g {virtualMachineRG} -n {worker['name']} || echo 'start {worker['name']} error') &"
        res = subprocess.run(start_bin, shell=True, capture_output=True)

        if res.returncode:
            output(output_level.RED, "start cluster failed!")
            raise("!!!")
        else:
            output(output_level.WHITE, "start cluster success!")

def stop_cluster():
    with open(f'~/clusterInfo.json', 'r') as infile:
        exist_cluster_info = json.load(infile)
    
    output(output_level.YELLOW, "Resource Group:", end="")
    output(output_level.WHITE, exist_cluster_info["virtualMachineRG"])

    output(output_level.YELLOW, "Machines to start:")
    for worker in exist_cluster_info["virtualMachines"]:
        output(output_level.WHITE, worker["name"])

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
    virtualMachineRG = exist_cluster_info["virtualMachineRG"]
    if answers["firstAffirm"] == "y" and answers["secondAffirm"] == "y":
        for worker in exist_cluster_info["virtualMachines"]:
            start_bin = f"(az vm deallocate -g {virtualMachineRG} -n {worker['name']} || echo 'stop {worker['name']} error') &"
        res = subprocess.run(start_bin, shell=True, capture_output=True)

        if res.returncode:
            output(output_level.RED, "stop cluster failed!")
            raise("!!!")
        else:
            output(output_level.WHITE, "stop cluster success!")

# def install_docker():

# unit test
if __name__ == "__main__":
    # with open(f'./azure_template/godconfig.json', 'r') as infile:
    #     exist_cluster_info = json.loads(infile.read())
    #     print(type(exist_cluster_info["virtualMachines"]))
    # res = subprocess.run("python azure_template/vmconfig.py azure_template/god_info_maro-demo.json azure_template/config_maro-demo azure_template/parameters.json", shell=True, capture_output=True)
    # print(res.returncode)
    # print("123")
    # create_god()
    password = subprocess.run(''' az storage account keys list --resource-group maro_dist --account-name dist4222797289618520747 --query "[0].value" | tr -d '"' ''', shell=True, capture_output=True)
    print(password.stdout)