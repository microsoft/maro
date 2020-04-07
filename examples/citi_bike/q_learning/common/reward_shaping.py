# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import defaultdict

import numpy as np

from maro.utils import Logger, LogFormat

class RewardShaping:
    def __init__(self, env, log_folder: str = './', log_enable: bool = True):
        self._env = env
        # self._station_idx2name = self._env.node_name_mapping
        self._station_idx2name = {key:key for key in self._env.agent_idx_list}
        self._agent_idx_list = self._env.agent_idx_list
        self._cache = {self._station_idx2name[agent_idx]: defaultdict(list) for agent_idx in self._agent_idx_list}
        self._log_enable = log_enable

        if self._log_enable:
            self._choose_action_logger_dict = {}
            for agent_idx in self._agent_idx_list:
                self._choose_action_logger_dict[self._station_idx2name[agent_idx]] = Logger(tag=f'{self._station_idx2name[agent_idx]}.choose_action',
                                                                                format_=LogFormat.none,
                                                                                dump_folder=log_folder, dump_mode='w', extension_name='csv',
                                                                                auto_timestamp=False)
                self._choose_action_logger_dict[self._station_idx2name[agent_idx]].debug(
                    ','.join(['episode', 'learning_index', 'tick', 'vessel_name', 'max_load', 'max_discharge', 'eps', 'port_states', 'vessel_states', 'action', 'actual_action', 'reward']))

    def calculate_reward(self, agent_name: str, current_ep: int, learning_rate: float):
        return NotImplemented

    def push_matrices(self, agent_name: str, matrix: dict):
        """
        store the cache from agents
        """
        for name, cache in matrix.items():
            self._cache[agent_name][name].append(cache)

    def _align_cache_by_next_state(self, agent_name: str):
        cache = self._cache[agent_name]

        cache['next_state'] = cache['state'][1:]
        # align
        for name, cache in cache.items():
            if name != 'next_state' and cache:
                cache.pop()

    def pop_experience(self, agent_name: str):
        """
        return the experience, and clear the cache
        """
        cache = self._cache[agent_name]
        experience_set = {name: cache[name] for name in ['state', 'action', 'reward', 'next_state', 'actual_action']}
        experience_set['info'] = [{'td_error': 1e10} for _ in range(len(cache['state']))]
        self._cache[agent_name] = defaultdict(list)
        return experience_set

    def _dump_logs(self, agent_name, current_ep, learning_rate):
        extra = ['eps', 'port_states', 'vessel_states', 'action', 'actual_action', 'reward']
        cache = self._cache[agent_name]
        event_list = cache['decision_event']
        for i in range(len(event_list)):
            event = event_list[i]
            max_load = str(event.action_scope.load)
            max_discharge = str(event.action_scope.discharge)
            log_str = ','.join([str(current_ep), str(learning_rate), str(event.tick), self._vessel_idx2name[event.vessel_idx], max_load, max_discharge])
            for name in extra:
                if type(cache[name][i]) == np.ndarray:
                    log_str += ',' + ','.join([str(f) for f in cache[name][i]])
                else:
                    log_str += ',' + str(cache[name][i])
            self._choose_action_logger_dict[agent_name].debug(log_str)


class TruncateReward(RewardShaping):
    def __init__(self, env, agent_idx_list: [int], offset: int = 100, reward_factor: float = 1.0, cost_factor: float = 1.0,
                 shortage_factor: float = 1.0, time_decay: float = 0.97, log_folder: str = './', log_enable: bool = False):
        super().__init__(env, log_folder=log_folder, log_enable=log_enable)
        self._agent_idx_list = agent_idx_list
        self._offset = offset
        self._reward_factor = reward_factor
        self._shortage_factor = shortage_factor
        self._cost_factor = cost_factor
        self._time_decay_factor = time_decay

    def calculate_reward(self, agent_name, current_ep, learning_rate):
        cache = self._cache[agent_name]
        snapshot_list = self._env.snapshot_list
        for i, tick in enumerate(cache['action_tick']):
            start_tick = tick + 1
            end_tick = tick + self._offset
            
            #calculate tc reward
            decay_list = [self._time_decay_factor ** i for i in range(end_tick - start_tick)
                      for _ in range(len(self._agent_idx_list))]

            
            fulfillments = snapshot_list.static_nodes[list(range(start_tick, end_tick)):self._agent_idx_list:('fulfillment', 0)]
            tot_fulfillment = np.dot(fulfillments, decay_list)

            shortages = snapshot_list.static_nodes[list(range(start_tick, end_tick)):self._agent_idx_list:("shortage", 0)]
            tot_shortage = np.dot(shortages, decay_list)

            costs = snapshot_list.static_nodes[list(range(start_tick, end_tick)):self._agent_idx_list:("extra_cost", 0)]
            tot_cost = np.dot(costs, decay_list)

            cache['reward'].append(np.float32(self._reward_factor * (tot_fulfillment - self._shortage_factor * tot_shortage
                                              - self._cost_factor * tot_cost)))

        self._align_cache_by_next_state(agent_name)

        if self._log_enable:
            self._dump_logs(agent_name, current_ep, learning_rate)