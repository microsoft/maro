# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import defaultdict

import numpy as np

from maro.simulator.graph import SnapshotList, ResourceNodeType


def truncate_reward(snapshot_list: SnapshotList, agent_idx_list: [int], start_tick: int, end_tick: int, fulfillment_factor: float = 1.0, shortage_factor: float = 1.0, time_decay: float = 0.97) -> np.float32:
    decay_list = [time_decay ** i for i in range(end_tick - start_tick) for j in range(len(agent_idx_list))]
    tot_fulfillment = np.dot(snapshot_list.get_attributes(
                ResourceNodeType.STATIC, [i for i in range(start_tick, end_tick)], agent_idx_list, ['fulfillment'], [0]), decay_list)
    tot_shortage = np.dot(snapshot_list.get_attributes(
                ResourceNodeType.STATIC, [i for i in range(start_tick, end_tick)], agent_idx_list, ['shortage'], [0]), decay_list)

    return np.float32(fulfillment_factor * tot_fulfillment - shortage_factor * tot_shortage)


def golden_finger_reward(topology, port_name: str, vessel_name: str, action_space: [float],
                         action_index: int, base: int = 1, gamma: float = 0.5) -> np.float32:
    '''
    For 4p_ssdd_simple, the best action is:
       supply_port_001: load 70% for route 1 and 30% for route 2
       supply_port_002: load 100%,
       demand_port_001: discharge 50%,
       demand_port_002: discharge 100%,

    For 5p_ssddd_simple, the best action is:
       transfer_port_001: discharge 100% on route_001, load 50% on route_002
       supply_port_001: load 100%
       supply_port_002: load 100%
       demand_port_001: discharge 50%
       demand_port_002: discharge 100%

    For 6p_sssbdd_simple, the best action is:
       transfer_port_001: load 100% on route_002, discharge 100% on route_003
       transfer_port_002: load 100% on route_003, discharge 100% on route_001
       supply_port_001: load 100%
       supply_port_002: load 100%
       demand_port_001: discharge 50%
       demand_port_002: discharge 100%
    '''
    action2index = {v: i for i, v in enumerate(action_space)}
    if topology.startswith('4p_ssdd'):
        best_action_idx_dict = {
            'supply_port_001': action2index[-0.5] if vessel_name.startswith('rt1') else action2index[-0.5],
            'supply_port_002': action2index[-1.0],
            'demand_port_001': action2index[1.0],
            'demand_port_002': action2index[1.0]
        }
    elif topology.startswith('5p_ssddd'):
        best_action_idx_dict = {
            'transfer_port_001': action2index[1.0] if vessel_name.startswith('rt1') else action2index[-0.5],
            'supply_port_001': action2index[-1.0],
            'supply_port_002': action2index[-1.0],
            'demand_port_001': action2index[0.5],
            'demand_port_002': action2index[1.0]
        }
    elif topology.startswith('6p_sssbdd'):
        best_action_idx_dict = {
            'transfer_port_001': action2index[-0.5] if vessel_name.startswith('rt2') else action2index[1.0],
            'transfer_port_002': action2index[-0.7] if vessel_name.startswith('rt3') else action2index[1.0],
            'supply_port_001': action2index[-1.0],
            'supply_port_002': action2index[-1.0],
            'demand_port_001': action2index[0.5],
            'demand_port_002': action2index[1.0]
        }
    else:
        raise ValueError('Unsupported topology')

    return np.float32(gamma ** abs(best_action_idx_dict[port_name] - action_index) * abs(base))


if __name__ == "__main__":
    res_cache = defaultdict(list)
    action_space = [round(i*0.1, 1) for i in range(-10, 11)]

    for port_name in ['transfer_port_001', 'supply_port_001', 'supply_port_002', 'demand_port_001', 'demand_port_002']:
        for i in range(len(action_space)):
            res_cache[port_name].append(golden_finger_reward('5p_ssddd', port_name, 'rt1_vessel_001',
                                                             action_space, i, 10))

    print('route_001:', res_cache)

    res_cache = defaultdict(list)
    for port_name in ['transfer_port_001', 'supply_port_001', 'supply_port_002', 'demand_port_001', 'demand_port_002']:
        for i in range(len(action_space)):
            res_cache[port_name].append(golden_finger_reward('5p_ssddd', port_name, 'rt2_vessel_001',
                                                             action_space, i, 10))

    print('route_002:', res_cache)
