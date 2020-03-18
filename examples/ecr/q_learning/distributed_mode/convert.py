# third party lib
import io
import os
import yaml
import random, string
import copy

CONFIG_PATH = os.environ.get('CONFIG') or 'config.yml'

with io.open(CONFIG_PATH, 'r') as in_file:
    config = yaml.safe_load(in_file)

if not os.path.exists('job_component_config'):
    os.mkdir('job_component_config')

def dump_config(component_config):
    file_name = f"{component_config['self_id']}.yml"
    with io.open(os.path.join('job_component_config', file_name), 'w', encoding='utf8') as out_file:
        yaml.safe_dump(component_config, out_file)

def convert():
    dist_config = config['distributed']

    group_name = config['experiment_name']

    if dist_config['auto_signature']:
        group_name += '_' + ''.join(random.sample(string.ascii_letters + string.digits, 6))

    component_list = ['environment_runner', 'learner']

    component_id_list_dict = dict()
    for component_type in component_list:
        component_id_list_dict[component_type] = \
            [component_type + '_' + str(i).zfill(3) for i in range(int(dist_config[component_type]['num']))]

    for component_type in component_list:
        peers_id_list = []
        for peer_name in component_list:
            if peer_name != component_type:
                peers_id_list.extend(component_id_list_dict[peer_name])

        for component_id in component_id_list_dict[component_type]:
            component_config = copy.deepcopy(config)
            component_config.pop('distributed')

            component_config['self_id'] = component_id
            component_config['peers_id'] = peers_id_list
            component_config['group_name'] = group_name
            component_config['resources'] = dist_config['resources'][component_type]
            component_config['redis'] = config['redis']
        
            dump_config(component_config)


if __name__ == "__main__":
    convert()

            
        
    


