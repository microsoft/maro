# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from datetime import datetime
import os
import sys
import random

import numpy as np
import torch
from tqdm import tqdm

from maro.utils import SimpleExperiencePool, Logger, LogFormat
from maro.simulator.scenarios.bike.common import Action, DecisionEvent


class Agent(object):
    def __init__(self, agent_name, algorithm, experience_pool: SimpleExperiencePool,
                 state_shaping, action_shaping, reward_shaping,
                 batch_num, batch_size, min_train_experience_num,
                 log_folder: str = None,
                 dashboard: object = None):
        self._agent_name = agent_name
        self._algorithm = algorithm
        self._experience_pool = experience_pool
        self._state_shaping = state_shaping
        self._action_shaping = action_shaping
        self._reward_shaping = reward_shaping
        self._batch_size = batch_size
        self._batch_num = batch_num
        self._min_train_experience_num = min_train_experience_num
        self._log_enable = False if log_folder is None else True
        self._dashboard = dashboard

        if self._log_enable:
            self._logger = Logger(tag='agent', format_=LogFormat.simple,
                                  dump_folder=log_folder, dump_mode='w', auto_timestamp=False)

    def store_experience(self, current_ep: int):
        self._reward_shaping.calculate_reward(self._agent_name, current_ep, self._algorithm.learning_index)
        experience_set = self._reward_shaping.pop_experience(self._agent_name)
        self._experience_pool.put(category_data_batches=[(name, cache) for name, cache in experience_set.items()])
        if self._log_enable:
            experience_summary = {name: experience_set[name] for name in ['action', 'actual_action', 'reward']}
            self._logger.debug(f'Agent {self._agent_name} new appended exp: {experience_summary}')
            self._logger.debug(f'Agent {self._agent_name} current experience pool size: {self._experience_pool.size}')

    def get_experience(self, current_ep: int):
        """
        return the experience from reward shaping. Only used for distributed mode
        """
        self._reward_shaping.calculate_reward(self._agent_name, current_ep, self._algorithm.learning_index)
        return self._reward_shaping.pop_experience(self._agent_name)

    @property
    def experience_pool(self):
        return self._experience_pool

    @property
    def algorithm(self):
        return self._algorithm

    def train(self, current_ep: int):
        """
        Args:
            current_ep (int): Current episode, which is used for logging.
        """
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
                np.array([sample_dict['action']])).permute(2,1,0)#.view(2, -1, 1)
            reward_batch = torch.from_numpy(
                np.array(sample_dict['reward'])).view(-1, 1)
            next_state_batch = torch.from_numpy(
                np.array(sample_dict['next_state'])).view(-1, self._algorithm.policy_net.input_dim)
            loss = self._algorithm.learn(state_batch=state_batch, action_batch=action_batch,
                                         reward_batch=reward_batch, next_state_batch=next_state_batch,
                                         current_ep=current_ep)

            # update td-error
            new_info_list = []
            for i in range(len(idx_list)):
                new_info_list.append({'td_error': loss})

            self._experience_pool.update([('info', idx_list, new_info_list)])

            if self._log_enable:
                self._logger.info(f'{self._agent_name} learn loss: {loss}')

            if self._dashboard is not None:
                self._dashboard.upload_exp_data(fields={self._agent_name: loss}, ep=current_ep, tick=None, measurement='bike_loss')

    def dump_modules(self, module_path: str):
        self._logger.debug(f'{self._agent_name} dump module to {module_path}')
        pass

    def load_modules(self, module_path: str):
        self._logger.debug(f'{self._agent_name} load module from {module_path}')
        pass

    def choose_action(self, decision_event: DecisionEvent, eps: float, current_ep: int, snapshot_list) -> Action:
        """
        Args:
            decision_event (DecisionEvent): Environment decision event, which includes the action scope, etc.
            snapshot_list: Environment snapshot.
            eps (float): Epsilon, which is used for exploration.
            current_ep (int): Current episode, which is used for logging.

        Returns:
            (Action): Environment action.
        """

        action_scope = decision_event.action_scope
        cur_tick = decision_event.tick
        cur_station_idx = decision_event.cell_idx
        cur_neighbor_idx_list = [int(x) for x in snapshot_list.static_nodes[0:cur_station_idx:("neighbors",[x for x in range(6)])]]
        
        numpy_state = self._state_shaping(
            cur_tick=cur_tick, cur_station_idx=cur_station_idx, cur_neighbor_idx_list= cur_neighbor_idx_list)

        state = torch.from_numpy(numpy_state).view(1, len(numpy_state))
        is_random, model_action = self._algorithm.choose_action(
            state=state, eps=eps, current_ep=current_ep, current_tick=cur_tick)

        neighbor_idx = cur_neighbor_idx_list[model_action[0]]
        neighbor_scope = action_scope[neighbor_idx] if neighbor_idx!= -1 else 0
        actual_action = self._action_shaping(action_idx= model_action[1], station_scope = action_scope[cur_station_idx],
                                            neighbor_scope = neighbor_scope)

        station_states = snapshot_list.static_nodes[
                      cur_tick: cur_station_idx: (['bikes', 'capacity', 'orders'], 0)]
        neighbor_states = snapshot_list.static_nodes[
                      cur_tick: neighbor_idx: (['bikes', 'capacity', 'orders'], 0)]
        self._reward_shaping.push_matrices(self._agent_name,
                                            {'state': numpy_state,
                                            'action': model_action,
                                            'actual_action': [neighbor_idx, actual_action],
                                            'action_tick': cur_tick,
                                            'decision_event': decision_event,
                                            'eps': eps,
                                            'station_states':station_states,
                                            'neighbor_states':neighbor_states})

        neighbor_idx = neighbor_idx if neighbor_idx!= -1 else cur_neighbor_idx_list[0]                                 
        env_action = Action(cur_station_idx, neighbor_idx, actual_action)
        if self._log_enable:
            self._logger.info(
                f'{self._agent_name} decision_event: {decision_event}, env_action: {env_action}, is_random: {is_random}')
        return env_action

    def load_policy_net_parameters(self, policy_net_parameters):
        """
        load updated policy net parameters to the algorithm.
        """
        self._algorithm.load_policy_net_parameters(policy_net_parameters)

    @property
    def experience_quantity(self):
        qty = self._experience_pool.size['action']
        return qty

    @property
    def model_size(self):
        size = sum([parameter.nelement() for parameter in self._algorithm.policy_net.parameters()])
        return size
