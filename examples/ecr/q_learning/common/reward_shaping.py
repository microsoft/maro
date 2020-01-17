# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import defaultdict

import numpy as np

from maro.simulator.graph import SnapshotList, ResourceNodeType

class RewardShaping():
    def __init__(self):
        self._reward_cache = []

    def __call__(self):
        pass

    @property
    def reward_cache(self):
        return self._reward_cache[:-1]

    def clear_cache(self):
        self._reward_cache = []

class TruncateReward(RewardShaping):
    def __init__(self, agent_idx_list: [int], fulfillment_factor: float = 1.0, shortage_factor: float = 1.0, time_decay: float = 0.97):
        super().__init__()
        self._agent_idx_list = agent_idx_list
        self._fulfillment_factor = fulfillment_factor
        self._shortage_factor = shortage_factor
        self._time_decay_factor = time_decay

    def __call__(self, snapshot_list: SnapshotList, start_tick: int, end_tick: int): 
        decay_list = [self._time_decay_factor ** i for i in range(end_tick - start_tick) for j in range(len(self._agent_idx_list))]
        tot_fulfillment = np.dot(snapshot_list.get_attributes(
                    ResourceNodeType.STATIC, [i for i in range(start_tick, end_tick)], self._agent_idx_list, ['fulfillment'], [0]), decay_list)
        tot_shortage = np.dot(snapshot_list.get_attributes(
                    ResourceNodeType.STATIC, [i for i in range(start_tick, end_tick)], self._agent_idx_list, ['shortage'], [0]), decay_list)

        self._reward_cache.append(np.float32(self._fulfillment_factor * tot_fulfillment - self._shortage_factor * tot_shortage))


class GoldenFingerReward(RewardShaping):
    def __init__(self, topology, action_space: [float], base: int = 1, gamma: float = 0.5):
        super().__init__()
        self._topology = topology
        self._action_space = action_space
        self._base = base
        self._gamma = gamma

    def __call__(self, port_name: str, vessel_name: str, action_index: int): 
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
        action2index = {v: i for i, v in enumerate(self._action_space)}
        if self._topology.startswith('4p_ssdd'):
            best_action_idx_dict = {
                'supply_port_001': action2index[-0.5] if vessel_name.startswith('rt1') else action2index[-0.5],
                'supply_port_002': action2index[-1.0],
                'demand_port_001': action2index[1.0],
                'demand_port_002': action2index[1.0]
            }
        elif self._topology.startswith('5p_ssddd'):
            best_action_idx_dict = {
                'transfer_port_001': action2index[1.0] if vessel_name.startswith('rt1') else action2index[-0.5],
                'supply_port_001': action2index[-1.0],
                'supply_port_002': action2index[-1.0],
                'demand_port_001': action2index[0.5],
                'demand_port_002': action2index[1.0]
            }
        elif self._topology.startswith('6p_sssbdd'):
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

        self._reward_cache.append(np.float32(self._gamma ** abs(best_action_idx_dict[port_name] - action_index) * abs(self._base)))


if __name__ == "__main__":
    action_space = [round(i*0.1, 1) for i in range(-10, 11)]
    topology = '5p_ssddd'
    a = GoldenFingerReward(topology, action_space)
    print(a.__class__.__name__, a.__class__.__name__ == "GoldenFingerReward")

    for port_name in ['transfer_port_001', 'supply_port_001', 'supply_port_002', 'demand_port_001', 'demand_port_002']:
        for i in range(len(action_space)):
            a(port_name, 'rt1_vessel_001', i)

    print('route_001:', a.reward_cache)

    a.clear_cache()
    for port_name in ['transfer_port_001', 'supply_port_001', 'supply_port_002', 'demand_port_001', 'demand_port_002']:
        for i in range(len(action_space)):
            a(port_name, 'rt2_vessel_001', i)

    print('route_002:', a.reward_cache)
    a.clear_cache()
    print(a.reward_cache)
