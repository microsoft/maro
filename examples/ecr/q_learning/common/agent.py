# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from datetime import datetime
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from maro.utils import SimpleExperiencePool, Logger, LogFormat
from maro.simulator.scenarios.ecr.common import Action, DecisionEvent


class Agent(object):
    def __init__(self, agent_name, topology, algorithm, experience_pool: SimpleExperiencePool,
                 state_shaping, action_shaping, reward_shaping,
                 batch_num, batch_size, min_train_experience_num,
                 log_enable: bool = True, log_folder: str = './',
                 dashboard_enable: bool = True, dashboard: object = None):
        self._agent_name = agent_name
        self._topology = topology
        self._algorithm = algorithm
        self._experience_pool = experience_pool
        self._state_shaping = state_shaping
        self._action_shaping = action_shaping
        self._reward_shaping = reward_shaping
        self._batch_size = batch_size
        self._batch_num = batch_num
        self._min_train_experience_num = min_train_experience_num
        self._log_enable = log_enable
        self._dashboard_enable = dashboard_enable
        self._dashboard = dashboard

        if self._log_enable:
            self._logger = Logger(tag='agent', format_=LogFormat.simple,
                                  dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
            self._choose_action_logger = Logger(tag=f'{self._algorithm.policy_net.name}.choose_action',
                                                format_=LogFormat.none,
                                                dump_folder=log_folder, dump_mode='w', extension_name='csv',
                                                auto_timestamp=False)

    def calculate_offline_rewards(self, current_ep: int):
        self._reward_shaping.update(self._agent_name)
        if self._log_enable:
            self._choose_action_logger.debug(f"episode {current_ep}, learning_index {self._algorithm.learning_index}:")
            extra = ['eps', 'port_states', 'vessel_states', 'action_index', 'actual_action', 'reward']
            self._choose_action_logger.debug(','.join(['tick', 'vessel_name', 'max_load', 'max_discharge'] + extra))
            for i in range(self._reward_shaping.get_event_count(self._agent_name)):
                log_str = self._reward_shaping.get_decision_event_info(self._agent_name, i, extra)
                self._choose_action_logger.debug(' '*10 + log_str)

    def store_experience(self):
        experience_set = self._reward_shaping.generate_experience_set(self._agent_name)
        self._experience_pool.put(category_data_batches=[(name, content) for name, content in experience_set.items()])
        if self._log_enable:
            experience_summary = {name: experience_set[name] for name in ['action_index', 'actual_action', 'reward']}
            self._logger.debug(f'Agent {self._agent_name} new appended exp: {experience_summary}')
            self._logger.debug(f'Agent {self._agent_name} current experience pool size: {self._experience_pool.size}')

        self._reward_shaping.clear_cache(self._agent_name)

    def get_experience(self):
        return self._reward_shaping.generate_experience_set(self._agent_name)

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
                ('action_index', idx_list),
                ('next_state', idx_list),
                ('info', idx_list)
            ])

            state_batch = torch.from_numpy(
                np.array(sample_dict['state'])).view(-1, self._algorithm.policy_net.input_dim)
            action_batch = torch.from_numpy(
                np.array(sample_dict['action_index'])).view(-1, 1)
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

            if self._dashboard_enable:
                self._dashboard.upload_loss({self._agent_name: loss}, current_ep)

    def dump_modules(self, module_path: str):
        self._logger.debug(f'{self._agent_name} dump module to {module_path}')
        pass

    def load_modules(self, module_path: str):
        self._logger.debug(f'{self._agent_name} load module from {module_path}')
        pass

    def choose_action(self, decision_event: DecisionEvent, eps: float, current_ep: int) -> Action:
        """
        Args:
            decision_event (DecisionEvent): Environment decision event, which includes the action scope, environment
            snapshot, etc.
            eps (float): Epsilon, which is used for exploration.
            current_ep (int): Current episode, which is used for logging.

        Returns:
            (Action): Environment action.
        """

        action_scope = decision_event.action_scope
        cur_tick = decision_event.tick
        cur_port_idx = decision_event.port_idx
        cur_vessel_idx = decision_event.vessel_idx
        snapshot_list = decision_event.snapshot_list

        numpy_state = self._state_shaping(
            cur_tick=cur_tick, cur_port_idx=cur_port_idx, cur_vessel_idx=cur_vessel_idx)

        state = torch.from_numpy(numpy_state).view(1, len(numpy_state))
        is_random, action_index = self._algorithm.choose_action(
            state=state, eps=eps, current_ep=current_ep)

        port_states = snapshot_list.static_nodes[
                      cur_tick: cur_port_idx: (['empty', 'full', 'on_shipper', 'on_consignee'], 0)]
        vessel_states = snapshot_list.dynamic_nodes[
                        cur_tick: cur_vessel_idx: (['empty', 'full', 'remaining_space'], 0)]
        early_discharge = snapshot_list.dynamic_nodes[
                        cur_tick: cur_vessel_idx: ('early_discharge', 0)][0]
        actual_action = self._action_shaping(scope=action_scope, action_index=action_index,
                                             port_empty=port_states[0], vessel_remaining_space=vessel_states[2],
                                             early_discharge=early_discharge)

        self._reward_shaping.fill_cache(self._agent_name,
                                        {'state': numpy_state,
                                         'action_index': action_index,
                                         'actual_action': actual_action,
                                         'action_tick': cur_tick,
                                         'decision_event': decision_event,
                                         'eps': eps,
                                         'port_states': port_states,
                                         'vessel_states': vessel_states})

        env_action = Action(cur_vessel_idx, cur_port_idx, actual_action)
        if self._log_enable:
            self._logger.info(
                f'{self._agent_name} decision_event: {decision_event}, env_action: {env_action}, is_random: {is_random}')
        return env_action

    def load_policy_net_parameters(self, policy_net_parameters):
        """
        load updated policy net parameters to the algorithm.
        """
        self._algorithm.load_policy_net_parameters(policy_net_parameters)
