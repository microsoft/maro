# third party lib
import io
import os
import yaml
import random, string
import copy

file_path = os.environ.get('NOMAD_CONFIG') or 'base_config/nomad_base.nomad'

def dump_config(nomad_name, nomad_cmd, nomad_args, cpu_req, memory_req):
    name_lines = ['job', 'group', 'task']
    new_lines = []
    
    with open(file_path, 'r+') as f:
        for line in f:
            new_lines.append(line)
    
    for l_idx, lines in enumerate(new_lines):
        for n in name_lines:
            if n in lines:
                idx = lines.index("\"")
                new_lines[l_idx] = lines[:idx] + "\"" + nomad_name + "\" {"
            
        if 'command' in lines:
            idx = lines.index("\"")
            new_lines[l_idx] = lines[:idx] + nomad_cmd
            
        if 'args' in lines:
            idx = lines.index('[')
            new_lines[l_idx] = lines[:idx] + str(nomad_args)
            
        if 'cpu' in lines:
            idx = lines.index('=')
            new_lines[l_idx] = lines[:idx+2] + cpu_req

        if 'memory' in lines:
            idx = lines.index('=')
            new_lines[l_idx] = lines[:idx+2] + memory_req

    file_name = f"{nomad_name}.nomad"
    with io.open(os.path.join('job_component_config', file_name), 'w', encoding='utf8') as out_file:
        for line in new_lines:
            out_file.write(line)
            if "\n" not in line:
                out_file.write("\n")


def nomad_job_generator(job_config):
    component_id = job_config['self_id']
    group_name = job_config['group_name']
    comp, idx = component_id.split('.')
    resource_config = job_config['resources']

    nomad_command = '\"python ' + str(comp) + '.py\"'
    nomad_args = ['LOG_LEVEL', 'PROGRESS', 'GROUP', str(group_name), 'COMPTYPE', str(comp), 'COMPID', str(idx)]
    cpu_req = resource_config['CPU']
    memory_req = resource_config['memory']

    dump_config(component_id, nomad_command, nomad_args, cpu_req, memory_req)