# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import defaultdict

import numpy as np

from maro.simulator.graph import SnapshotList, ResourceNodeType
from maro.utils import Logger, LogFormat

class RewardShaping:
    def __init__(self, env, log_folder: str = './', log_enable: bool = True):
        self._env = env
        self._cache = {}
        self._port_idx2name = self._env.node_name_mapping['static']
        self._vessel_idx2name = self._env.node_name_mapping['dynamic']
        self._agent_idx_list = self._env.agent_idx_list
        self._log_enable = log_enable

        if self._log_enable:
            self._choose_action_logger = {}
            for agent_idx in self._agent_idx_list:
                self._choose_action_logger[self._port_idx2name[agent_idx]] = Logger(tag=f'{self._port_idx2name[agent_idx]}.choose_action',
                                                                                format_=LogFormat.none,
                                                                                dump_folder=log_folder, dump_mode='w', extension_name='csv',
                                                                                auto_timestamp=False)

    def __call__(self, agent_name, current_ep, learning_rate):
        return NotImplemented

    def push(self, agent_name, content_dict):
        if agent_name not in self._cache:
            self._cache[agent_name] = defaultdict(list)
        for name, content in content_dict.items():
            self._cache[agent_name][name].append(content)

    def clear_cache(self, agent_name):
        for name, cache in self._cache[agent_name].items():
            cache.clear()

    def _align_cache_by_next_state(self, agent_name):
        cache = self._cache[agent_name]

        cache['next_state'] = cache['state'][1:]
        # align
        for name, cache in cache.items():
            if name != 'next_state':
                cache.pop()

    def generate_experience(self, agent_name):
        cache = self._cache[agent_name].copy()
        experience_set = {name: cache[name] for name in ['state', 'action', 'reward', 'next_state', 'actual_action']}
        experience_set['info'] = [{'td_error': 1e10} for _ in range(len(cache['state']))]
        return experience_set

    def _get_log_info(self, agent_name, idx, extra):
        cache = self._cache[agent_name]
        event = cache['decision_event'][idx]
        max_load = str(event.action_scope.load)
        max_discharge = str(event.action_scope.discharge)
        return ','.join([str(event.tick), self._vessel_idx2name[event.vessel_idx], max_load, max_discharge] +
                        [','.join([str(f) for f in cache[name][idx]]) if type(cache[name][idx]) == list else str(cache[name][idx])
                         for name in extra])

    def _record_logs(self, agent_name, current_ep, learning_rate):
        self._choose_action_logger[agent_name].debug(f"episode {current_ep}, learning_index {learning_rate}:")
        extra = ['eps', 'port_states', 'vessel_states', 'action', 'actual_action', 'reward']
        self._choose_action_logger[agent_name].debug(','.join(['tick', 'vessel_name', 'max_load', 'max_discharge'] + extra))
        for i in range(len(self._cache[agent_name]['decision_event'])):
            log_str = self._get_log_info(agent_name, i, extra)
            self._choose_action_logger[agent_name].debug(' '*10 + log_str)


class TruncateReward(RewardShaping):
    def __init__(self, env, agent_idx_list: [int], fulfillment_factor: float = 1.0,
                 shortage_factor: float = 1.0, time_decay: float = 0.97, log_folder: str = './', log_enable: bool = True):
        super().__init__(env, log_folder=log_folder, log_enable=log_enable)
        self._agent_idx_list = agent_idx_list
        self._fulfillment_factor = fulfillment_factor
        self._shortage_factor = shortage_factor
        self._time_decay_factor = time_decay

    def __call__(self, agent_name, current_ep, learning_rate):
        cache = self._cache[agent_name]
        snapshot_list = self._env.snapshot_list
        for i, tick in enumerate(cache['action_tick']):
            start_tick = tick + 1
            end_tick = tick + 100
            
            #calculate tc reward
            decay_list = [self._time_decay_factor ** i for i in range(end_tick - start_tick)
                      for _ in range(len(self._agent_idx_list))]
            tot_fulfillment = np.dot(snapshot_list.get_attributes(ResourceNodeType.STATIC, list(range(start_tick, end_tick)),
                                                            self._agent_idx_list, ['fulfillment'], [0]), decay_list)
            tot_shortage = np.dot(snapshot_list.get_attributes(ResourceNodeType.STATIC, list(range(start_tick, end_tick)),
                                                            self._agent_idx_list, ['shortage'], [0]), decay_list)

            cache['reward'].append(np.float32(self._fulfillment_factor * tot_fulfillment - self._shortage_factor * tot_shortage))

        self._align_cache_by_next_state(agent_name)

        if self._log_enable:
            self._record_logs(agent_name, current_ep, learning_rate)


class GoldenFingerReward(RewardShaping):
    def __init__(self, env, topology, action_space: [float],
                 base: int = 1, gamma: float = 0.5, log_folder: str = './', log_enable: bool = True):
        super().__init__(env, log_folder=log_folder, log_enable=log_enable)
        self._topology = topology
        self._action_space = action_space
        self._base = base
        self._gamma = gamma

    def __call__(self, agent_name, current_ep, learning_rate):
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
        cache = self._cache[agent_name]
        for i, tick in enumerate(cache['action_tick']):
            port_name = self._port_idx2name[cache['decision_event'][i].port_idx]
            vessel_name = self._vessel_idx2name[cache['decision_event'][i].vessel_idx]
            action_index = cache['action'][i]

            # calculate gf rewards
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

            cache['reward'].append(np.float32(self._gamma ** abs(best_action_idx_dict[port_name] - action_index) * abs(self._base)))

        self._align_cache_by_next_state(agent_name)

        if self._log_enable:
            self._record_logs(agent_name, current_ep, learning_rate)
        
        
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
