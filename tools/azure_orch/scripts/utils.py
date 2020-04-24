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

def sync_log():
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



# rsync -arvz  --exclude="log/*" src/ dest/ --delete

# rsync -avrz --include="log" --include="log/*" --exclude="*" dest/ src/
