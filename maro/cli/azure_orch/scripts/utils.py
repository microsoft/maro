import time
import logging
import os
import subprocess
import inquirer
import json
import socket
import getpass
import chalk

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from maro.cli.azure_orch.scripts.gen_job_config import gen_job_config

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('logger')

def sync_code():
    questions = [
        inquirer.Text(
            'project_path',
            message="Where is your project located on dev machine?",
            default=os.environ['PYTHONPATH'],
        ),
        inquirer.List(
            'resource_group_name', 
            message="Which resource group do you want to sync?",
            choices=[resource_group_info[:-5] for resource_group_info in os.listdir(f"/maro/dist/azure_template/resource_group_info")],
            carousel=True,
        ),
        inquirer.Text(
            'branch_name',
            message="What is the name of the branch you want to sync?"
        ),
        
    ]

    answers = inquirer.prompt(questions)

    resource_group_name = answers['resource_group_name']
    branch_name = answers['branch_name']
    project_path = answers['project_path']

    with open(f'/maro/dist/azure_template/resource_group_info/{resource_group_name}.json', 'r') as infile:
        resource_group_info = json.load(infile)

    admin_username = resource_group_info['admin_username']
    god_IP = resource_group_info['virtual_machines'][0]['IP']
    
    mkdir_bin = f"ssh -o StrictHostKeyChecking=no {admin_username}@{god_IP} 'sudo mkdir /code_repo/{branch_name}; sudo chmod -R 777 /code_repo/{branch_name}'"
    sync_bin = f"rsync -arvz --exclude='log/*' --exclude='job_config/*' {project_path} {admin_username}@{god_IP}:/code_repo/{branch_name} --delete"

    for bin in [mkdir_bin, sync_bin]:
        res = subprocess.run(bin, shell=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info(f"run {bin} success!")

def pull_log():
    questions = [
        inquirer.Text(
            'project_path',
            message="Where is your project located on dev machine?",
            default=os.environ['PYTHONPATH'],
        ),
        inquirer.List(
            'resource_group_name', 
            message="Which resource group do you want to sync?",
            choices=[resource_group_info[:-5] for resource_group_info in os.listdir(f"/maro/dist/azure_template/resource_group_info")],
            carousel=True,
        ),
        inquirer.Text(
            'branch_name',
            message="what is the name of the branch you want to sync?"
        ),
    ]

    answers = inquirer.prompt(questions)

    resource_group_name = answers['resource_group_name']
    branch_name = answers['branch_name']
    project_path = answers['project_path']

    with open(f'/maro/dist/azure_template/resource_group_info/{resource_group_name}.json', 'r') as infile:
        resource_group_info = json.load(infile)
    
    admin_username = resource_group_info['admin_username']
    god_IP = resource_group_info['virtual_machines'][0]['IP']
    
    sync_bin = f"rsync -arvz --include='log' --include='log/*' --include='log/*/*' --include='log/*/*/*' --exclude='*' {admin_username}@{god_IP}:/code_repo/{branch_name}/maro/ {project_path}"

    res = subprocess.run(sync_bin, shell=True)
    if res.returncode:
        raise Exception(res.stderr)
    else:
        logger.info(f"run {sync_bin} success!")

def generate_job_config():
    questions = [
        inquirer.Text(
            'config_path', 
            message="Where is your meta config file?",
            default=os.path.join(os.environ['PYTHONPATH'], "examples/ecr/q_learning/distributed_mode/config.yml"),
        ),
    ]

    config_path = inquirer.prompt(questions)['config_path']

    if socket.gethostname() == 'god':
        questions = [
            inquirer.Text(
                'branch_name',
                message="What is the name of your branch?",
            )
        ]

        out_folder = f"/code_repo/{inquirer.prompt(questions)['branch_name']}/maro"
    else:
        out_folder = "/maro/dist/"

    job_group_name = gen_job_config(config_path, out_folder)

    if socket.gethostname() != 'god':
        questions = [
            inquirer.List(
                'resource_group_name',
                message="Which resource group would you like to launch this job?",
                choices=[resource_group_name[:-5] for resource_group_name in os.listdir("/maro/dist/azure_template/resource_group_info")],
                carousel=True,
            ),
            inquirer.Text(
                'branch_name',
                message="What is the name of your branch?",
            )
        ]

        answers = inquirer.prompt(questions)
        resource_group_name = answers['resource_group_name']
        branch_name = answers['branch_name']

        with open(f'/maro/dist/azure_template/resource_group_info/{resource_group_name}.json') as infile:
            resource_group_info = json.load(infile)
        
        admin_username = resource_group_info['admin_username']
        god_IP = resource_group_info['virtual_machines'][0]['IP']

        rsync_bin = f"rsync -arvz -r /maro/dist/job_config/{job_group_name} {admin_username}@{god_IP}:/code_repo/{branch_name}/maro/job_config"

        res = subprocess.run(rsync_bin, shell=True, capture_output=True)
        if res.returncode:
            raise Exception(res.stderr)
        else:
            logger.info(f"run {rsync_bin} success!")
    
    logger.info(chalk.green(f"your job group name is {job_group_name}, please remember it!"))

def sync_resource_group_info():
    if not os.path.exists('/maro/dist/azure_template/resource_group_info'):
        os.mkdir('/maro/dist/azure_template/resource_group_info')

    questions = [
        inquirer.Text(
            'admin_username', 
            message="What is the admin username on god?",
        ),
        inquirer.Text(
            'god_IP', 
            message="What is the IP address of god?",
        ),
        inquirer.Text(
            'public_key',
            message="What is your public key? (default is ~/.ssh/id_rsa.pub)",
            default=open(f'/home/{getpass.getuser()}/.ssh/id_rsa.pub').read()
        ),
    ]

    answers = inquirer.prompt(questions)
    admin_username = answers['admin_username']
    god_IP = answers['god_IP']
    public_key = answers['public_key']

    logger.critical(chalk.red("please make sure that you have added your public key on the god machine!"))
    logger.info(f"you are syncing the resource group to you local machine with the god: {admin_username}@{god_IP}")
    scp_bin = f"scp -o StrictHostKeyChecking=no {admin_username}@{god_IP}:~/resource_group_info.json /maro/dist/azure_template/resource_group_info/ "
    res = subprocess.run(scp_bin, shell=True, capture_output=True)
    if res.returncode:
        raise Exception(res.stderr)
    else:
        logger.info(f"sync resource group info with {admin_username}@{god_IP} success!")
    
    with open('/maro/dist/azure_template/resource_group_info/resource_group_info.json') as infile:
        resource_group_info = json.load(infile)

    for node in resource_group_info['virtual_machines']:
        if node['name'] != 'god':
            pubkey_bin = f'''ssh -o StrictHostKeyChecking=no {admin_username}@{god_IP} "ssh -o StrictHostKeyChecking=no {admin_username}@{node['IP']} 'echo {public_key} >> ~/.ssh/authorized_keys'"'''
            res = subprocess.run(pubkey_bin, shell=True, capture_output=True)
            if res.returncode:
                raise Exception(res.stderr)
    
    resource_group_name = resource_group_info['virtual_machine_resource_group']
    os.rename('/maro/dist/azure_template/resource_group_info/resource_group_info.json', f'/maro/dist/azure_template/resource_group_info/{resource_group_name}.json')

# auto sync your code to the your branch on the code repo of god when you do any modifications
# maybe unstable, use it with cautiousness!
def dev_mode():
    questions = [
        inquirer.Text(
            'project_path',
            message="Where is your project located on dev machine?",
            default=os.environ['PYTHONPATH'],
        ),
        inquirer.List(
            'resource_group_name',
            message="Which resource group would you like to deloy code?",
            choices=[resource_group_name[:-5] for resource_group_name in os.listdir("/maro/dist/azure_template/resource_group_info")],
            carousel=True,
        ),
        inquirer.Text(
            'branch_name',
            message="What is the name of your branch?",
        )
    ]

    answers = inquirer.prompt(questions)
    project_path = answers['project_path']
    resource_group_name = answers['resource_group_name']
    branch_name = answers['branch_name']

    with open(f'/maro/dist/azure_template/resource_group_info/{resource_group_name}.json', 'r') as infile:
        resource_group_info = json.load(infile)

    admin_username = resource_group_info['admin_username']
    god_IP = resource_group_info['virtual_machines'][0]['IP']

    code_observer = Observer()
    code_event_handler = CodeFileEventHandler(f"rsync -arvz --exclude='log/*' --exclude='job_config/*' {project_path} {admin_username}@{god_IP}:/code_repo/{branch_name} --delete", resource_group_name, branch_name)
    code_observer.schedule(code_event_handler, project_path, True)
    code_observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        code_observer.stop()

    code_observer.join()

class CodeFileEventHandler(FileSystemEventHandler):
    def __init__(self, auto_sync_bin, resource_group_name, branch_name):
        FileSystemEventHandler.__init__(self)
        self._auto_sync_bin = auto_sync_bin
        self._resource_group_name = resource_group_name
        self._branch_name = branch_name
        
    def on_moved(self, event):
        subprocess.run(self._auto_sync_bin, shell=True, capture_output=True)
        logging.info(f"[move] from {event.src_path} to {event.dest_path} sync to: {self._resource_group_name}:{self._branch_name}")

    def on_created(self, event):
        subprocess.run(self._auto_sync_bin, shell=True, capture_output=True)
        logging.info(f"[create] {event.src_path} sync to: {self._resource_group_name}:{self._branch_name}")


    def on_deleted(self, event):
        subprocess.run(self._auto_sync_bin, shell=True, capture_output=True)
        logging.info(f"[delete] {event.src_path} sync to: {self._resource_group_name}:{self._branch_name}")


    def on_modified(self, event):
        subprocess.run(self._auto_sync_bin, shell=True, capture_output=True)
        logging.info(f"[modify] {event.src_path} sync to: {self._resource_group_name}:{self._branch_name}")