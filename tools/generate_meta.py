# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

SCHEDULE_NAME = 'demo_config_offline_apply_window_size'
ALGORITHM = 'offline_lp'
WORK_DIR = '/maro_dev/examples/ecr/lp/offline_lp'
BASE_CONFIG = '/home/jinywan/maro/examples/ecr/lp/offline_lp/config.yml'
DOCKER_IMAGE_NAME = 'github_pulp'
PARALLEL_NUM = 18

def write_init(NAME):
    meta_file = open(f'schedule_meta/{NAME}.meta.yml', 'w')
    meta_file.write(f'name: {NAME}\n')
    meta_file.write('mail_to: dummy@microsoft.com\n')
    meta_file.write(f'base_config: {BASE_CONFIG}\n')
    meta_file.write('auto_notification: false\n')
    meta_file.write(f'docker_image: {DOCKER_IMAGE_NAME}\n')
    meta_file.write(f'parallel_num: {PARALLEL_NUM}\n')
    meta_file.write('jobs:\n')

    return meta_file

def write_for_config(meta_file, experiment_name, config_name, apply_size, delay_factor):
    meta_file.write(f'  {experiment_name}:\n')
    meta_file.write(f'    parameters:\n')
    meta_file.write(f'      experiment_name: {experiment_name}\n')
    meta_file.write(f'      env.topology: {config_name}\n')
    meta_file.write(f'      lp.params.apply_buffer_size: {apply_size}\n')
    meta_file.write(f'      lp.factor.full_delayed_factor: {delay_factor}\n')
    meta_file.write(f'    work_dir: {WORK_DIR}\n')
    meta_file.write(f'    run_cmd: \"python runner.py\"\n')

if __name__ == "__main__":
    meta_file = write_init(NAME =SCHEDULE_NAME)

    for apply_size in [1, 10, 20, 40, 60, 80]:
        for delay_factor in [0, 0.1, 0.01, 0.001]:
            for topology in ["5p_ssddd"]:
                for level in ['A', 'D', 'J', 'P']:
                    experiment_name = f'{ALGORITHM}_{topology}_{level}_{apply_size}_{delay_factor}'
                    config_name = f'{topology}_{level}'
                    write_for_config(meta_file, experiment_name, config_name, apply_size, delay_factor)

    meta_file.close()
