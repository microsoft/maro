# third party lib
import io
import os
import yaml
import random, string
import copy

K8S_CONFIG_PATH = os.environ.get('K8S_CONFIG') or 'base_config/k8s_base.yml'

with io.open(K8S_CONFIG_PATH, 'r') as in_file:
    k8s_config = yaml.safe_load(in_file)

def dump_config(component_config):
    file_name = f"{component_config['metadata']['name']}.k8s.yml"
    with io.open(os.path.join('job_component_config', file_name), 'w', encoding='utf8') as out_file:
        yaml.safe_dump(component_config, out_file)

def k8s_job_generator(job_config):
    # k8s config
    component_id = job_config['self_id']
    group_name = job_config['group_name']
    comp, idx = component_id.split('.')
    resource_config = job_config['resources']

    k8s_config['metadata']['name'] = component_id

    k8s_config_containers = k8s_config['spec']['template']['spec']['containers']
    k8s_config_containers['name'] = component_id
    k8s_config_containers['resources']['requests']['cpu'] = resource_config['CPU']
    k8s_config_containers['resources']['requests']['memory'] = resource_config['memory']
    
    job_command = ['LOG_LEVEL=PROGRESS', 'GROUP='+str(group_name), 'COMPTYPE='+str(comp), 'COMPID='+str(idx),'python', str(comp)+'.py']
    k8s_config_containers['command'] = job_command

    #dump k8s job config
    dump_config(k8s_config)
