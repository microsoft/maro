# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
from math import floor

def get_available_envs():
    """
    get available environment settings under scenarios folder

    Returns:
        list of environment settings like [{"scenario": "ecr", "topology": "5p_ssddd"}]
    """
    root_foler = os.path.join(os.path.split(os.path.realpath(__file__))[0], "../scenarios")
    topologies_folder = "topologies"

    envs = []

    _, scenarios, _ = next(os.walk(root_foler))

    for scenario in scenarios:
        try:
            _, topologies, _ = next(os.walk(f'{root_foler}/{scenario}/{topologies_folder}'))
        except StopIteration:
            # continue if there is no sub folder(topology)
            continue

        for topology in topologies:
            envs.append({'scenario': scenario, 'topology': topology})

    return envs

def tick_to_frame_index(start_tick: int, cur_tick: int, resolution: int):
    return floor((cur_tick - start_tick)/resolution)
