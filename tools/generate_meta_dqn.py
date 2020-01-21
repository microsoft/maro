# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

MAX_TICK = 224
SCHEDULE_NAME = 'config_adjustment_single_host_gf_VCRR_2.5'
ALGORITHMS = ['tc'] # 'tc' 'lp'
WORK_DIR = '/maro_dev/examples/ecr/q_learning/single_host_mode'  # '/maro_dev/tools/replay_lp'
BASE_CONFIG = '/home/starmie/maro/examples/ecr/q_learning/single_host_mode/config.yml'
DOCKER_IMAGE_NAME = 'maro/dev'
PARALLEL_NUM = 4
SEEDS = [0]

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

def write_for_config(meta_file, experiment_name, config_name, algorithm, seed):
    meta_file.write(f'  {experiment_name}:\n')
    meta_file.write(f'    parameters:\n')
    meta_file.write(f'      experiment_name: {experiment_name}\n')
    meta_file.write(f'      env.topology: {config_name}\n')
    meta_file.write(f'      env.max_tick: {MAX_TICK}\n')
    meta_file.write(f'      train.reward_shaping: {algorithm}\n')
    meta_file.write(f'      train.seed: {seed}\n')
    meta_file.write(f'      train.num_threads: {int(64 / PARALLEL_NUM)}\n')
    meta_file.write(f'    work_dir: {WORK_DIR}\n')
    meta_file.write(f'    run_cmd: \"python runner.py\"\n')

if __name__ == "__main__":
    meta_file = write_init(NAME =SCHEDULE_NAME)

    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            for topology in ["4p_ssdd", "5p_ssddd", "6p_sssbdd"]: #, "22p_global_ratio"]:
                for level in range(9):
                    experiment_name = f'{topology}_{algorithm}_{MAX_TICK}_l0.{level}_{SCHEDULE_NAME[-8:]}'
                    config_name = f'{topology}_l0.{level}'
                    write_for_config(meta_file, experiment_name, config_name, algorithm, seed)

    meta_file.close()
