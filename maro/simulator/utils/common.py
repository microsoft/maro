# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
from typing import List
from math import floor, ceil

topologies_folder = "topologies"
scenarios_root_folder = os.path.join(os.path.split(os.path.realpath(__file__))[0], "../scenarios")

def get_available_envs():
    """
    get available environment settings under scenarios folder

    Returns:
        List[dict]: list of environment settings like [{"scenario": "ecr", "topology": "5p_ssddd"}]
    """
    

    envs = []

    scenarios = get_scenarios()

    for scenario in scenarios:
        for topology in get_topologies(scenario):
            envs.append({'scenario': scenario, 'topology': topology})

    return envs

def get_scenarios()->List[str]:
    """Get built-in scenario names
    
    Returns:
        List[str]: list of scenario name
    """
    try:
        _, scenarios, _ = next(os.walk(scenarios_root_folder))
    except StopIteration:
        return []

    return [s for s in scenarios if not s.startswith("__")]

def get_topologies(scenario: str)-> List[str]:
    """Get topology list of built-in scenario

    Args:
        scenario(str): built-in scenario name

    Return:
        List[str]: list of topology name
    """
    scenario_topology_root = f'{scenarios_root_folder}/{scenario}/{topologies_folder}'

    if not os.path.exists(scenario_topology_root):
        return []

    try:
        _, topologies, _ = next(os.walk(scenario_topology_root))
    except StopIteration:
        return []

    return topologies

def tick_to_frame_index(start_tick: int, cur_tick: int, resolution: int) -> int:
    """Calculate frame index in snapshot list
    
    Args:
        start_tick(int): start tick of current simulation
        cur_tick(int): current tick in simulator
        resolution(int): snapshot resolution

    Returns:
        int: frame index in snapshot list of current tick
    """
    return floor((cur_tick - start_tick)/resolution)

def frame_index_to_ticks(start_tick: int, max_tick: int, resolution:int) -> dict:
    """Return frame index to ticks mapping dictionary"""
    
    mapping = {}

    max_snapshot_num = total_frames(start_tick, max_tick, resolution)


    for frame_index in range(max_snapshot_num):
        frame_start_tick = start_tick + frame_index * resolution
        frame_end_tick = min(max_tick, frame_start_tick + resolution)

        mapping[frame_index] = [t for t in range(frame_start_tick, frame_end_tick)]

    return mapping


def total_frames(start_tick: int, max_tick: int, resolution: int) -> int:
    """Calculate total frames in snapshot
    
    Args:
        start_tick(int): start tick of current simulation
        max_tick(int): max tick of current simulation
        resolution(int): snapshot resolution
    
    Returns:
        int: max snapshots will be generated, NOTE: this is the max, but you can use small value to reduce memory using
    """
    return ceil((max_tick - start_tick)/resolution)

__all__ = ['get_available_envs', 'get_scenarios', 'get_topologies', 'tick_to_frame_index', 'frame_index_to_ticks', 'total_frames']
