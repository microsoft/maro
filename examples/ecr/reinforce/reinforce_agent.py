from datetime import datetime
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from examples.ecr.common.reward_shaping import TruncateReward, GoldenFingerReward, SelfAwareTruncatedReward
from maro.simulator.scenarios.ecr.common import Action, DecisionEvent
from maro.utils import SimpleExperiencePool, Logger, LogFormat

class Agent(object):
    def __init__(self, agent_name, port_idx2name,
                vessel_idx2name, algorithm, experience_pool: SimpleExperiencePool,
                state_shaping, action_shaping, reward_shaping, batch_num, batch_size, min_train_experience_num,
                topology: str, agent_idx_list: list, agent_idx: int,
                log_enable: bool = True, log_folder: str = './',
                dashboard_enable: bool = True, dashboard: object = None):
        self._agent_name = agent_name
        self._port_idx2name = port_idx2name
        self._vessel_idx2name = vessel_idx2name
        self._algorithm = algorithm
        self._experience_pool = experience_pool
        self._state_shaping = state_shaping
        self._action_shaping = action_shaping
        self._reward_shaping = reward_shaping
        self._state_cache = []
        self._action_cache = []
        self._action_tick_cache = []
        self._actual_action_cache = []
        self._reward_cache = []
        self._next_state_cache = []
        self._decision_event_cache = []
        self._port_states_cache = []
        self._vessel_states_cache = []
        self._batch_size = batch_size
        self._batch_num = batch_num
        self._min_train_experience_num = min_train_experience_num
        self._log_enable = log_enable
        self._topology = topology

        if reward_shaping == 'gf':
            self._reward_shaping = GoldenFingerReward(topology=self._topology, action_space=self._action_shaping.action_space, base=10)
        elif reward_shaping == 'tc':
            self._reward_shaping = TruncateReward(agent_idx_list=agent_idx_list)
        elif reward_shaping == 'stc':
            self._reward_shaping = SelfAwareTruncatedReward(agent_idx_list=agent_idx_list, agent_idx=agent_idx)
        else:
            raise KeyError(f'Invalid reward shaping: {reward_shaping}')

        if self._log_enable:
            self._logger = Logger(tag='agent', 
                                dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
            self._choose_action_logger = Logger(tag=f'{self._algorithm.policy_net.name}.choose_action',
                                                dump_folder=log_folder, dump_mode='w', extension_name='csv', auto_timestamp=False)
            self._choose_action_logger.debug('episode,tick,learning_index,port_empty,port_full,port_on_shipper,port_on_consignee,vessel_empty,vessel_full,vessel_remaining_space,max_load_num,max_discharge_num,vessel_name,action_index,actual_action,reward')

    def fulfill_cache(self, agent_idx_list: [int], snapshot_list, current_ep: int):
        for i, tick in enumerate(self._action_tick_cache):
            if type(self._reward_shaping) == GoldenFingerReward:
                self._reward_shaping(port_name=self._port_idx2name[self._decision_event_cache[i].port_idx],
                                              vessel_name=self._vessel_idx2name[self._decision_event_cache[i].vessel_idx],
                                              action_index=self._action_cache[i])
            elif type(self._reward_shaping) == TruncateReward:
                self._reward_shaping(snapshot_list=snapshot_list, start_tick=tick + 1, end_tick=tick + 100)
            elif type(self._reward_shaping) == SelfAwareTruncatedReward:
                self._reward_shaping(snapshot_list=snapshot_list, start_tick=tick + 1, end_tick=tick + 100)
            else:
                raise KeyError(f'Unknown reward shaping type: {type(self._reward_shaping)}')

        self._reward_cache = self._reward_shaping.reward_cache
        self._action_tick_cache = []
        self._next_state_cache = self._state_cache[1:]
        self._state_cache = self._state_cache[:-1]
        self._action_cache = self._action_cache[:-1]
        self._actual_action_cache = self._actual_action_cache[:-1]
        self._decision_event_cache = self._decision_event_cache[:-1]
        self._port_states_cache = self._port_states_cache[:-1]
        self._vessel_states_cache = self._vessel_states_cache[:-1]
        if self._log_enable:
            self._logger.debug(f'Agent {self._agent_name} current experience pool size: {self._experience_pool.size}')
        
        exp_summary = [{'action': action, 'actual_action': actual_action, 'reward': reward}
                        for action, actual_action, reward in zip(self._action_cache, self._actual_action_cache, self._reward_cache)]

        if self._log_enable:
            self._logger.debug(f'Agent {self._agent_name} new appended exp: {exp_summary}')
            for i, decision_event in enumerate(self._decision_event_cache):
                episode = current_ep
                tick = decision_event.tick
                learning_index = self._algorithm.learning_index
                port_states = self._port_states_cache[i]
                vessel_states = self._vessel_states_cache[i]
                max_load_num = self._decision_event_cache[i].action_scope.load
                max_discharge_num = self._decision_event_cache[i].action_scope.discharge
                vessel_name = self._vessel_idx2name[self._decision_event_cache[i].vessel_idx]
                action_index = self._action_cache[i]
                actual_action = self._actual_action_cache[i]
                reward = self._reward_cache[i]
                log_str = f"{episode},{tick},{learning_index},{','.join([str(f) for f in port_states])},{','.join([str(f) for f in vessel_states])},{max_load_num},{max_discharge_num},{vessel_name},{action_index},{actual_action},{reward}"
                self._choose_action_logger.debug(log_str)

    def put_experience(self):
        self._experience_pool.put(category_data_batches=[
            ('state', self._state_cache), ('action', self._action_cache),
            ('reward', self._reward_cache), ('next_state', self._next_state_cache),
            ('actual_action', self._actual_action_cache),
            ('info', [{'td_error': 1e10}
                      for i in range(len(self._state_cache))])
        ])

    def clear_cache(self):
        self._next_state_cache = []
        self._state_cache = []
        self._reward_cache = []
        self._action_cache = []
        self._actual_action_cache = []
        self._decision_event_cache = []
        self._eps_cache = []
        self._port_states_cache = []
        self._vessel_states_cache = []
        
    @property
    def experience_pool(self):
        return self._experience_pool

    def experience_pool_reset(self):
        self._experience_pool.reset()

    def train(self, current_ep:int):
        '''
        Args:
            current_ep (int): Current episode, which is used for logging.
        '''
        # TODO: add per-agent min experience
        if self._experience_pool.size['info'] < self._min_train_experience_num:
            return 0

        pbar = tqdm(range(self._batch_num))
        for i in pbar:
            pbar.set_description(f'Agent {self._agent_name} batch training')
            idx_list = self._experience_pool.apply_multi_samplers(
                category_samplers=[('info', [(lambda i, o: (i, o['td_error']), self._batch_size)])])['info']
            sample_dict = self._experience_pool.get(category_idx_batches=[
                ('state', idx_list),
                ('reward', idx_list),
                ('action', idx_list),
                ('next_state', idx_list),
                ('info', idx_list)
            ])

            state_batch = torch.from_numpy(
                np.array(sample_dict['state'])).view(-1, self._algorithm.policy_net.input_dim)
            action_batch = torch.from_numpy(
                np.array(sample_dict['action'])).view(-1, 1)
            reward_batch = torch.from_numpy(
                np.array(sample_dict['reward'])).view(-1, 1)
            next_state_batch = torch.from_numpy(
                np.array(sample_dict['next_state'])).view(-1, self._algorithm.policy_net.input_dim)
            loss = self._algorithm.learn(state_batch=state_batch, action_batch=action_batch,
                                  reward_batch=reward_batch, next_state_batch=next_state_batch, current_ep=current_ep)

            # update td-error
            new_info_list = []
            for i in range(len(idx_list)):
                new_info_list.append({'td_error': loss})

            self._experience_pool.update([('info', idx_list, new_info_list)])

            if self._log_enable:
                self._logger.info(f'{self._agent_name} learn loss: {loss}')

    def dump_policy_net_parameters(self, dump_path):
        """
        dump policy net parameters to given path.
        """
        self._algorithm.dump_policy_net_parameters(dump_path)
        
    def dump_modules(self, module_path: str):
        self._logger.debug(f'{self._agent_name} dump module to {module_path}')
        pass

    def load_modules(self, module_path: str):
        self._logger.debug(f'{self._agent_name} load module from {module_path}')
        pass

    def choose_action(self, decision_event: DecisionEvent, eps: float, current_ep: int) -> Action:
        '''
        Args:
            decision_event (DecisionEvent): Environment decision event, which includes the action scope, environment snapshot, etc.
            eps (float): Epsilon, which is used for exploration.
            current_ep (int): Current episode, which is used for logging.

        Returns:
            (Action): Environment action.
        '''

        action_scope = decision_event.action_scope
        cur_tick = decision_event.tick
        cur_port_idx = decision_event.port_idx
        cur_vessel_idx = decision_event.vessel_idx
        snapshot_list = decision_event.snapshot_list

        numpy_state = self._state_shaping(
            cur_tick=cur_tick, cur_port_idx=cur_port_idx, cur_vessel_idx=cur_vessel_idx)

        state = torch.from_numpy(numpy_state).view(1, len(numpy_state))
        action_index = self._algorithm.choose_action(
            state=state, current_ep=current_ep)

        self._state_cache.append(numpy_state)
        self._action_cache.append(action_index)
        self._action_tick_cache.append(cur_tick)
        self._decision_event_cache.append(decision_event)
        port_states = snapshot_list.static_nodes[cur_tick: [cur_port_idx]: (['empty', 'full', 'on_shipper', 'on_consignee', 'booking'], 0)]
        vessel_states = snapshot_list.dynamic_nodes[cur_tick: [cur_vessel_idx]: (['empty', 'full', 'remaining_space'], 0)]
        early_discharge = snapshot_list.dynamic_nodes[cur_tick: cur_vessel_idx: ('early_discharge', 0)][0]
        # port_states = snapshot_list.static_nodes[cur_tick: [cur_port_idx]: (['empty', 'full', 'on_shipper', 'on_consignee'], 0)]
        # vessel_states = snapshot_list.dynamic_nodes[cur_tick: [cur_vessel_idx]: (['empty', 'full', 'remaining_space'], 0)]
        self._port_states_cache.append(port_states)
        self._vessel_states_cache.append(vessel_states)
        actual_action = self._action_shaping(scope=action_scope, action_index=action_index, port_empty=port_states[0],
                                             vessel_remaining_space=vessel_states[2], early_discharge=early_discharge)
        self._actual_action_cache.append(actual_action)
        env_action = Action(cur_vessel_idx, cur_port_idx, actual_action)
        if self._log_enable:
            self._logger.info(f'{self._agent_name} decision_event: {decision_event}, env_action: {env_action}')
        return env_action

        
