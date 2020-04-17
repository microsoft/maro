# third party lib
import io
import os
import yaml
import random, string
import copy


K8S_CONFIG_PATH = os.environ.get('K8S_CONFIG') or 'base_config/base_k8s.yml'

with io.open(K8S_CONFIG_PATH, 'r') as in_file:
    k8s_config = yaml.safe_load(in_file)

CONFIG_PATH = os.environ.get('CONFIG') or 'config.yml'

with io.open(CONFIG_PATH, 'r') as in_file:
    config = yaml.safe_load(in_file)

if not os.path.exists('job_component_config'):
    os.mkdir('job_component_config')

def dump_config(component_config, self_id):
    file_name = str(self_id)+".k8s.yml"
    with io.open(os.path.join('job_component_config', file_name), 'w', encoding='utf8') as out_file:
        yaml.safe_dump(component_config, out_file)

def k8s_job_generator(job_config, group_name, component_id):
    # k8s config
    index = component_id.rindex("_")
    comp, idx = component_id[:index], component_id[index+1:]

    k8s_name = component_id.replace('_', '-')
    k8s_config['metadata']['name'] = k8s_name

    k8s_config_containers = k8s_config['spec']['template']['spec']['containers'][0]    
    k8s_config_containers['name'] = k8s_name
    k8s_config_containers['resources']['requests']['cpu'] = job_config['CPU']
    k8s_config_containers['resources']['requests']['memory'] = job_config['memory']
    
    job_command = ['python', '/maro/examples/ecr/q_learning/distributed_mode/'+str(comp)+'.py']
    k8s_config_containers['command'] = job_command
    k8s_env = k8s_config_containers['env']
    config_path = '/maro/examples/ecr/q_learning/distributed_mode/config.yml'
    env_value = ['PROGRESS', group_name, comp, idx, config_path]
    for idx, env_set in enumerate(k8s_env):
        env_set['value'] = env_value[idx]

    #dump k8s job config
    dump_config(k8s_config, component_id)


def convert():
    dist_config = config['distributed']

    group_name = config['experiment_name']

    component_list = ['environment_runner', 'learner']

    component_id_list_dict = dict()
    for component_type in component_list:
        component_id_list_dict[component_type] = \
            [component_type + '_' + str(i) for i in range(int(dist_config[component_type]['num']))]

    for component_type in component_list:
        for component_id in component_id_list_dict[component_type]:
            component_config = copy.deepcopy(dist_config['resources'][component_type])

            # k8s config
            k8s_job_generator(component_config, group_name, component_id)          


if __name__ == "__main__":
    convert()