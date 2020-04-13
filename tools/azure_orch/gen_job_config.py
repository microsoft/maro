# third party lib
import io
import os
import yaml
import random, string
import copy

def dump_config(component_config, group_name):
    file_name = f"{component_config['self_id']}.yml"
    with io.open(os.path.join(f"job_config/{group_name}", file_name), 'w', encoding='utf8') as out_file:
        yaml.safe_dump(component_config, out_file)

def gen_job_config(config_path):
    with open(config_path, 'r') as infile:
        config = yaml.safe_load(infile)

    dist_config = config['distributed']

    group_name = config['experiment_name']

    if dist_config['auto_signature']:
        group_name += '_' + ''.join(random.sample(string.ascii_letters + string.digits, 6))

    if not os.path.exists('job_config'):
        os.mkdir('job_config')
    if not os.path.exists(f"job_config/{group_name}"):
        os.mkdir(f"job_config/{group_name}")

    component_list = ['environment_runner', 'learner']

    component_id_list_dict = dict()
    for component_type in component_list:
        component_id_list_dict[component_type] = \
            [component_type + '.' + str(i) for i in range(int(dist_config[component_type]['num']))]
    try:
        for component_type in component_list:
            peers_id_list = []
            for peer_name in component_list:
                if peer_name != component_type:
                    peers_id_list.extend(component_id_list_dict[peer_name])

            for component_id in component_id_list_dict[component_type]:
                component_config = copy.deepcopy(config)
                component_config.pop('distributed')

                component_config['self_id'] = component_id
                component_config['self_component_type'] = component_type
                component_config['peers_id'] = peers_id_list
                component_config['group_name'] = group_name
                component_config['resources'] = dist_config['resources'][component_type]
                component_config['redis'] = config['redis']
            
                dump_config(component_config, group_name)
    except BaseException as e:
        os.removedirs(f"job_config/{group_name}")
        raise(e)

            
        
    


