# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
from math import ceil, floor
from typing import List

topologies_folder = "topologies"
scenarios_root_folder = os.path.join(os.path.split(os.path.realpath(__file__))[0], "../scenarios")


def get_available_envs():
    """Get available built-in scenarios and their topologies.

    Returns:
        List[dict]: List of environment settings like [{"scenario": "cim", "topology": "toy.5p_ssddd_l0.1"}].
    """
    envs = []

    scenarios = get_scenarios()

    for scenario in scenarios:
        for topology in get_topologies(scenario):
            envs.append({'scenario': scenario, 'topology': topology})

    return envs


def get_scenarios() -> List[str]:
    """Get built-in scenario name list.

    Returns:
        List[str]: List of scenario name.
    """
    try:
        _, scenarios, _ = next(os.walk(scenarios_root_folder))
        scenarios = sorted([s for s in scenarios if not s.startswith("__")])

    except StopIteration:
        return []

    return scenarios


def get_topologies(scenario: str) -> List[str]:
    """Get topology list of specified built-in scenario name.

    Args:
        scenario(str): Built-in scenario name.

    Return:
        List[str]: List of topology name.
    """
    scenario_topology_root = f'{scenarios_root_folder}/{scenario}/{topologies_folder}'

    if not os.path.exists(scenario_topology_root):
        return []

    try:
        _, topologies, _ = next(os.walk(scenario_topology_root))
        topologies = sorted(topologies)

    except StopIteration:
        return []

    return topologies


def tick_to_frame_index(start_tick: int, cur_tick: int, resolution: int) -> int:
    """Calculate frame index in snapshot list of specified configurations, usually is used
    when taking snapshot.

    Args:
        start_tick(int): Start tick of current simulation.
        cur_tick(int): Current tick in simulator.
        resolution(int): Snapshot resolution.

    Returns:
        int: Frame index in snapshot list of current tick.
    """
    return floor((cur_tick - start_tick) / resolution)


def frame_index_to_ticks(start_tick: int, max_tick: int, resolution: int) -> dict:
    """Calculate a dictionary that key is frame index, value is ticks.

    Args:
        start_tick (int): Start tick of current simulation.
        max_tick (int): Max tick of current simulation.
        resolution (int): Current snapshot resolution.

    Returns:
        dict: Key is the frame index in snapshot list, value is tick list for current frame index.
    """
    mapping = {}

    max_snapshot_num = total_frames(start_tick, max_tick, resolution)

    for frame_index in range(max_snapshot_num):
        frame_start_tick = start_tick + frame_index * resolution
        frame_end_tick = min(max_tick, frame_start_tick + resolution)

        mapping[frame_index] = [t for t in range(frame_start_tick, frame_end_tick)]

    return mapping


def total_frames(start_tick: int, max_tick: int, resolution: int) -> int:
    """Calculate total frame snapshot in snapshot list.

    NOTE:
        This method return the max snapshot number,
        but you can use small value to reduce memory using your own one.

    Args:
        start_tick(int): Start tick of current simulation.
        max_tick(int): Max tick of current simulation.
        resolution(int): Snapshot resolution.

    Returns:
        int: Max snapshots will be generated.
    """
    return ceil((max_tick - start_tick) / resolution)


__all__ = ['get_available_envs', 'get_scenarios', 'get_topologies',
           'tick_to_frame_index', 'frame_index_to_ticks', 'total_frames']
