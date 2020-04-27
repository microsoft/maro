import time
import logging
import os
import subprocess
import inquirer
import json

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('logger')

def sync_code():
    questions = [
        inquirer.List(
            'resourceGroupName', 
            message="Which resource group do you want to sync?",
            choices=[resource_group_info[:-5] for resource_group_info in os.listdir(f"azure_template/resource_group_info")],
            carousel=True,
        ),
        inquirer.Text(
            'codebaseName',
            message="What is the name of the codebase you want to sync?"
        ),
    ]

    answers = inquirer.prompt(questions)

    resource_group_name = answers['resourceGroupName']
    codebase_name = answers['codebaseName']

    with open(f'azure_template/resource_group_info/{resource_group_name}.json', 'r') as infile:
        resource_group_info = json.load(infile)
    
    project_dir = os.environ['PYTHONPATH']

    admin_username = resource_group_info['adminUsername']
    god_IP = resource_group_info['virtualMachines'][0]['IP']
    
    mkdir_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{god_IP} 'sudo mkdir /code_point/{codebase_name}; sudo chmod -R 777 /code_point/{codebase_name}'"
    sync_bin = f"rsync -arvz --exclude='log/*' --exclude='.git/*' {project_dir} {admin_username}@{god_IP}:/code_point/{codebase_name} --delete"

    for bin in [mkdir_bin, sync_bin]:
        res = subprocess.run(bin, shell=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info(f"run {bin} success!")

def pull_log():
    questions = [
        inquirer.List(
            'resourceGroupName', 
            message="Which resource group do you want to sync?",
            choices=[resource_group_info[:-5] for resource_group_info in os.listdir(f"azure_template/resource_group_info")],
            carousel=True,
        ),
        inquirer.Text(
            'codebaseName',
            message="what is the name of the codebase you want to sync?"
        ),
    ]

    answers = inquirer.prompt(questions)

    resource_group_name = answers['resourceGroupName']
    codebase_name = answers['codebaseName']

    with open(f'azure_template/resource_group_info/{resource_group_name}.json', 'r') as infile:
        resource_group_info = json.load(infile)
    
    project_dir = os.environ['PYTHONPATH']

    admin_username = resource_group_info['adminUsername']
    god_IP = resource_group_info['virtualMachines'][0]['IP']
    
    sync_bin = f"rsync -arvz --include='log' --include='log/*' --include='log/*/*' --include='log/*/*/*' --exclude='*' {admin_username}@{god_IP}:/code_point/{codebase_name}/maro/ {project_dir}"

    res = subprocess.run(sync_bin, shell=True)
    if res.returncode:
        raise Exception(res.stderr)
    else:
        logger.info(f"run {sync_bin} success!")

def generate_job_config():
    PYTHONPATH = os.environ['PYTHONPATH']
    default_config_path = os.path.join(PYTHONPATH, "examples/ecr/q_learning/distributed_mode/config.yml")

    questions = [
        inquirer.Text(
            'configPath', 
            message="Where is your meta config file?",
            default=default_config_path
        ),
    ]

    config_path = inquirer.prompt(questions)['configPath']

    job_group_name = gen_job_config(config_path)

    if socket.gethostname() != 'god':
        questions = [
            inquirer.List(
                'resourceGroupName',
                message="Which resource group would you like to launch job?",
                choices=[resource_group_name[:-5] for resource_group_name in os.listdir("azure_template/resource_group_info")],
                carousel=True,
            ),
            inquirer.Text(
                'codebaseName',
                message="What is the name of your codebase?",
            )
        ]

        answers = inquirer.prompt(questions)
        resource_group_name = answers['resourceGroupName']
        codebase_name = answers['codebaseName']

        with open(f'azure_template/resource_group_info/{resource_group_name}.json') as infile:
            resource_group_info = json.load(infile)
        
        admin_username = resource_group_info['adminUsername']
        god_IP = resource_group_info['virtualMachines'][0]['IP']

        rsync_bin = f"rsync -arvz -r job_config/{job_group_name} {admin_username}@{god_IP}:/code_point/{codebase_name}/maro/tools/azure_orch/job_config"

        res = subprocess.run(rsync_bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info(f"run {rsync_bin} success!")

def sync_resource_group_info():
    if not os.path.exists('azure_template/resource_group_info'):
        os.mkdir('azure_template/resource_group_info')

    questions = [
        inquirer.Text(
            'adminUsername', 
            message="What is the admin username on god?",
        ),
        inquirer.Text(
            'godIP', 
            message="What is the IP address of god?",
        ),      
    ]

    god_info = inquirer.prompt(questions)
    admin_username = god_info['adminUsername']
    god_IP = god_info['godIP']

    logger.critical(chalk.red("please make sure that you have added your public key on the god machine!"))
    logger.info(f"you are syncing the resource group to you local machine with the god: {admin_username}@{god_IP}")
    scp_bin = f"scp -o StrictHostKeyChecking=no {admin_username}@{god_IP}:~/resource_group_info.json azure_template/resource_group_info/ "
    res = subprocess.run(scp_bin, shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)
    else:
        logger.info(f"sync resource group info with {admin_username}@{god_IP} success!")


def deploy_code():
    logger.critical(chalk.red("if you are not lucy, please make sure that you have sync with the resource group you want to deploy code on!"))

    project_dir = os.environ['PYTHONPATH']
    questions = [
        inquirer.List(
            'resourceGroupName',
            message="Which resource group would you like to deloy code?",
            choices=[resource_group_name[:-5] for resource_group_name in os.listdir("azure_template/resource_group_info")],
            carousel=True,
        ),
        inquirer.Text(
            'codebaseName',
            message="What is the name of your codebase?",
        )
    ]

    deploy_info = inquirer.prompt(questions)

    resource_group_name = deploy_info['resourceGroupName']
    codebase_name = deploy_info['codebaseName']

    with open(f'azure_template/resource_group_info/{resource_group_name}.json', 'r') as infile:
        resource_group_info = json.load(infile)
    
    admin_username = resource_group_info['adminUsername']
    god_IP = resource_group_info['virtualMachines'][0]['IP']
    
    mkdir_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{god_IP} 'sudo mkdir /code_point/{codebase_name}; sudo chmod -R 777 /code_point/{codebase_name}'"
    deploy_bin = f"rsync -arvz --exclude='log/*' -r {project_dir} {admin_username}@{god_IP}:/code_point/{codebase_name}"

    for bin in [mkdir_bin, deploy_bin]:
        res = subprocess.run(bin, shell=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info(f"run {bin} success!")


# rsync -arvz  --exclude="log/*" src/ dest/ --delete

# rsync -avrz --include="log" --include="log/*" --exclude="*" dest/ src/


# class CodeFileEventHandler(FileSystemEventHandler):
#     def __init__(self, src, dest):
#         FileSystemEventHandler.__init__(self)
#         self._src = src
#         self._dest = dest
        
#     def on_moved(self, event):
#         sync(self._src, self._dest, 'sync', purge=True)
#         logging.info("[move] from {0} to {1} sync to: {3}".format(event.src_path, event.dest_path, self._dest))

#     def on_created(self, event):
#         sync(self._src, self._dest, 'sync', purge=True)
#         logging.info("[create] {0} sync to: {1}".format(event.src_path, self._dest))


#     def on_deleted(self, event):
#         sync(self._src, self._dest, 'sync', purge=True)
#         logging.info("[delete] {0} sync to: {1}".format(event.src_path, self._dest))


#     def on_modified(self, event):
#         sync(self._src, self._dest, 'sync', purge=True)
#         logging.info("[modify] {0} sync to: {1}".format(event.src_path, self._dest))

# class LogFileEventHandler(FileSystemEventHandler):
#     def __init__(self, src, dest):
#         FileSystemEventHandler.__init__(self)
#         self._src = src
#         self._dest = dest

#     def on_created(self, event):
#         sync(self._src, self._dest, 'sync')
#         logging.info("[create] {0} sync to: {1}".format(event.src_path, self._dest))
    
#     def on_modified(self, event):
#         sync(self._src, self._dest, 'sync')
#         logging.info("[modify] {0} sync to: {1}".format(event.src_path, self._dest))


# def auto_sync(src, dest):
#     code_observer = Observer()
#     # log_observer = Observer()
#     code_event_handler = CodeFileEventHandler(src, dest)
#     # log_event_handler = LogFileEventHandler(dest, src)
#     code_observer.schedule(code_event_handler, src, True)
#     # log_observer.schedule(log_event_handler, dest, True)
#     code_observer.start()
#     # log_observer.start()
    
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         code_observer.stop()
#         # log_observer.stop()

#     code_observer.join()
#     # log_observer.join()
