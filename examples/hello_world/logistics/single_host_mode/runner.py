# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import io
import os
import random
import numpy as np
import torch
import yaml
import time

import maro.simulator.utils.random as sim_random
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
from maro.simulator import Env
from maro.utils import SimpleExperiencePool, Logger, LogFormat, convert_dottable

from examples.hello_world.logistics.q_learning.common.agent import Agent
from examples.hello_world.logistics.q_learning.common.dqn import QNet, DQN
from examples.hello_world.logistics.q_learning.common.reward_shaping import TruncateReward
from examples.hello_world.logistics.q_learning.common.state_shaping import StateShaping
from examples.hello_world.logistics.q_learning.common.action_shaping import DiscreteActionShaping


####################################################### START OF INITIAL_PARAMETERS #######################################################

CONFIG_PATH = os.environ.get('CONFIG') or 'config.yml'

with io.open(CONFIG_PATH, 'r') as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

LOG_FOLDER = os.path.join(os.getcwd(), 'log', f"{datetime.now().strftime('%Y%m%d')}", config.experiment_name)
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

with io.open(os.path.join(LOG_FOLDER, 'config.yml'), 'w', encoding='utf8') as out_file:
    yaml.safe_dump(raw_config, out_file)

EXPERIMENT_NAME = config.experiment_name
SCENARIO = config.env.scenario
TOPOLOGY = config.env.topology
MAX_TICK = config.env.max_tick
MAX_TRAIN_EP = config.train.max_ep
MAX_TEST_EP = config.test.max_ep
MAX_EPS = config.train.exploration.max_eps
PHASE_SPLIT_POINT = config.train.exploration.phase_split_point  # exploration two phase split point
FIRST_PHASE_REDUCE_PROPORTION = config.train.exploration.first_phase_reduce_proportion  # first phase reduce proportion of max_eps
TARGET_UPDATE_FREQ = config.train.dqn.target_update_frequency
LEARNING_RATE = config.train.dqn.lr
DROPOUT_P = config.train.dqn.dropout_p
GAMMA = config.train.dqn.gamma  # Reward decay
TAU = config.train.dqn.tau  # Soft update
BATCH_NUM = config.train.batch_num
BATCH_SIZE = config.train.batch_size
MIN_TRAIN_EXP_NUM = config.train.min_train_experience_num  # when experience num is less than this num, agent will not train model
TRAIN_SEED = config.train.seed
TEST_SEED = config.test.seed
QNET_SEED = config.qnet.seed
RUNNER_LOG_ENABLE = config.log.runner.enable
AGENT_LOG_ENABLE = config.log.agent.enable
DQN_LOG_ENABLE = config.log.dqn.enable
DQN_LOG_DROPOUT_P = config.log.dqn.dropout_p
QNET_LOG_ENABLE = config.log.qnet.enable

if config.train.reward_shaping not in {'tc'}:
    raise ValueError('Unsupported reward shaping. Currently supported reward shaping types: "tc"')

REWARD_SHAPING = config.train.reward_shaping

# Config for dashboard
DASHBOARD_ENABLE = config.dashboard.enable
DASHBOARD_LOG_ENABLE = config.log.dashboard.enable
DASHBOARD_HOST = config.dashboard.influxdb.host
DASHBOARD_PORT = config.dashboard.influxdb.port
DASHBOARD_USE_UDP = config.dashboard.influxdb.use_udp
DASHBOARD_UDP_PORT = config.dashboard.influxdb.udp_port
RANKLIST_ENABLE = config.dashboard.ranklist.enable
AUTHOR = config.dashboard.ranklist.author
COMMIT = config.dashboard.ranklist.commit


####################################################### END OF INITIAL_PARAMETERS #######################################################


class Runner:
    def __init__(self, scenario: str, topology: str, max_tick: int, max_train_ep: int, max_test_ep: int,
                 eps_list: [float], log_enable: bool = True):

        # Init for dashboard
        self._scenario = scenario
        self._topology = topology
        self._max_train_ep = max_train_ep
        self._max_test_ep = max_test_ep
        self._max_tick = max_tick
        self._eps_list = eps_list
        self._log_enable = log_enable
        self._set_seed(TRAIN_SEED)
        self._env = Env(scenario, topology, max_tick = max_tick)
        self._warehouse_idx2name = self._env.node_name_mapping['static']
        self._agent_dict = self._load_agent(self._env.agent_idx_list)
        self._warehouse_name2idx = {}
        for idx in self._warehouse_idx2name.keys():
            self._warehouse_name2idx[self._warehouse_idx2name[idx]] = idx
        self._time_cost = OrderedDict()
        if log_enable:
            self._logger = Logger(tag='runner', 
                                  format_=LogFormat.simple,
                                  dump_folder=LOG_FOLDER, 
                                  dump_mode='w', 
                                  auto_timestamp=False)
            self._performance_logger = Logger(tag=f'runner.performance', 
                                              format_=LogFormat.none,
                                              dump_folder=LOG_FOLDER, 
                                              dump_mode='w', 
                                              extension_name='csv',
                                              auto_timestamp=False)


    def _load_agent(self, agent_idx_list: [int]):
        self._dashboard = None
        self._set_seed(QNET_SEED)
        agent_dict = {}
        state_shaping = StateShaping(env=self._env,
                                     relative_tick_list=[-1, -2, -3, -4, -5, -6, -7],
                                     warehouse_attribute_list=['weekday', 'stock', 'demand', 'fulfilled', 'unfulfilled'])
        action_space = list(range(11))
        action_shaping = DiscreteActionShaping(action_space=action_space)
        
        if REWARD_SHAPING == 'tc':
            reward_shaping = TruncateReward(env=self._env, agent_idx_list=agent_idx_list, log_folder=LOG_FOLDER)
        else:
            raise ValueError('Unsupported Reward Shaping')

        for agent_idx in agent_idx_list:
            experience_pool = SimpleExperiencePool()
            policy_net = QNet(name=f'{self._warehouse_idx2name[agent_idx]}.policy', 
                              input_dim=state_shaping.dim,
                              hidden_dims=[256, 128, 64], 
                              output_dim=len(action_space), 
                              dropout_p=DROPOUT_P,
                              log_folder=LOG_FOLDER if QNET_LOG_ENABLE else None)
            target_net = QNet(name=f'{self._warehouse_idx2name[agent_idx]}.target', 
                              input_dim=state_shaping.dim,
                              hidden_dims=[256, 128, 64], 
                              output_dim=len(action_space), 
                              dropout_p=DROPOUT_P,
                              log_folder=LOG_FOLDER if QNET_LOG_ENABLE else None)
            target_net.load_state_dict(policy_net.state_dict())
            dqn = DQN(policy_net=policy_net, 
                      target_net=target_net,
                      gamma=GAMMA, 
                      tau=TAU, 
                      target_update_frequency=TARGET_UPDATE_FREQ, 
                      lr=LEARNING_RATE,
                      log_folder=LOG_FOLDER if DQN_LOG_ENABLE else None, 
                      log_dropout_p=DQN_LOG_DROPOUT_P,
                      dashboard=self._dashboard)
            agent_dict[agent_idx] = Agent(agent_name=self._warehouse_idx2name[agent_idx],
                                          algorithm=dqn, 
                                          experience_pool=experience_pool,
                                          state_shaping=state_shaping,
                                          action_shaping=action_shaping,
                                          reward_shaping=reward_shaping,
                                          batch_num=BATCH_NUM, 
                                          batch_size=BATCH_SIZE,
                                          min_train_experience_num=MIN_TRAIN_EXP_NUM,
                                          log_folder=LOG_FOLDER if AGENT_LOG_ENABLE else None,
                                          dashboard=self._dashboard)

        return agent_dict

    def start(self):
        pbar = tqdm(range(self._max_train_ep))

        for ep in pbar:
            time_dict = OrderedDict()
            ep_start = time.time()
            self._set_seed(TRAIN_SEED + ep)
            pbar.set_description('train episode')
            env_start = time.time()
            _, decision_event, is_done = self._env.step(None)

            while not is_done:
                action = self._agent_dict[decision_event.warehouse_idx].choose_action(
                        decision_event=decision_event, eps=self._eps_list[ep], current_ep=ep, snapshot_list= self._env.snapshot_list)
                _, decision_event, is_done = self._env.step(action)
                
            time_dict['env_time'] = time.time() - env_start
            time_dict['train_time'] = 0
            for agent in self._agent_dict.values():
                train_start = time.time()
                agent.store_experience(current_ep=ep)
                train_start = time.time()
                agent.train(current_ep=ep)
                time_dict[agent._agent_name] = time.time() - train_start
                time_dict['train_time'] += time_dict[agent._agent_name]

            if self._log_enable:
                self._print_summary(ep=ep, is_train=True)

            self._env.reset()

            time_dict['ep_time'] = time.time() - ep_start
            time_dict['other_time'] = time_dict['ep_time'] - time_dict['train_time'] - time_dict['env_time']
            self._time_cost[ep] = time_dict

        self._test()

    def _test(self):
        pbar = tqdm(range(self._max_test_ep))
        for ep in pbar:
            time_dict = OrderedDict()
            ep_start = time.time()
            self._set_seed(TEST_SEED)
            pbar.set_description('test episode')
            env_start = time.time()
            _, decision_event, is_done = self._env.step(None)
            tot_reward = 0
            while not is_done:
                action = self._agent_dict[decision_event.warehouse_idx].choose_action(
                    decision_event=decision_event, 
                    eps=self._eps_list[ep], 
                    current_ep=ep, 
                    snapshot_list=self._env.snapshot_list)
                reward, decision_event, is_done = self._env.step(action)
                tot_reward += reward or 0
        
            print("Cumulative Reward: {}".format(tot_reward))
            time_dict['env_time'] = time.time() - env_start
            if self._log_enable:
                self._print_summary(ep=ep, is_train=False)

            self._env.reset()

            time_dict['ep_time'] = time.time() - ep_start
            time_dict['other_time'] = time_dict['ep_time'] - time_dict['env_time']
            self._time_cost[ep + self._max_train_ep] = time_dict

    def _print_summary(self, ep, is_train: bool = True):
        stock_list = self._env.snapshot_list.static_nodes[
                        self._env.tick: self._env.agent_idx_list: ('stock', 0)]
        pretty_stock_dict = OrderedDict()
        tot_stock = 0
        for i, stock in enumerate(stock_list):
            pretty_stock_dict[self._warehouse_idx2name[i]] = stock
            tot_stock += stock
        pretty_stock_dict['total_stock'] = tot_stock

        demand_list = self._env.snapshot_list.static_nodes[
                       self._env.tick: self._env.agent_idx_list: ('demand', 0)]
        pretty_demand_dict = OrderedDict()
        tot_demand = 0
        for i, demand in enumerate(demand_list):
            pretty_demand_dict[self._warehouse_idx2name[i]] = demand
            tot_demand += demand
        pretty_demand_dict['total_demand'] = tot_demand

        if is_train:
            self._performance_logger.debug(
                f"{ep},{self._eps_list[ep]},{','.join([str(value) for value in pretty_demand_dict.values()])},{','.join([str(value) for value in pretty_stock_dict.values()])}")
            self._logger.critical(
                f'{self._env.name} | train | [{ep + 1}/{self._max_train_ep}] total tick: {self._max_tick}, total demand: {pretty_demand_dict}, total stock: {pretty_stock_dict}')
        else:
            self._logger.critical(
                f'{self._env.name} | test | [{ep + 1}/{self._max_test_ep}] total tick: {self._max_tick}, total demand: {pretty_demand_dict}, total stock: {pretty_stock_dict}')

        if self._dashboard is not None:
            self._send_to_dashboard(ep, pretty_stock_dict, pretty_demand_dict, is_train)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        sim_random.seed(seed)


if __name__ == '__main__':
    phase_split_point = PHASE_SPLIT_POINT
    first_phase_eps_delta = MAX_EPS * FIRST_PHASE_REDUCE_PROPORTION
    first_phase_total_ep = MAX_TRAIN_EP * phase_split_point
    second_phase_eps_delta = MAX_EPS * (1 - FIRST_PHASE_REDUCE_PROPORTION)
    second_phase_total_ep = MAX_TRAIN_EP * (1 - phase_split_point)

    first_phase_eps_step = first_phase_eps_delta / (first_phase_total_ep + 1e-10)
    second_phase_eps_step = second_phase_eps_delta / (second_phase_total_ep - 1 + 1e-10)

    eps_list = []
    for i in range(MAX_TRAIN_EP):
        if i < first_phase_total_ep:
            eps_list.append(MAX_EPS - i * first_phase_eps_step)
        else:
            eps_list.append(MAX_EPS - first_phase_eps_delta - (i - first_phase_total_ep) * second_phase_eps_step)

    eps_list[-1] = 0.0

    runner = Runner(scenario=SCENARIO, 
                    topology=TOPOLOGY,
                    max_tick=MAX_TICK, 
                    max_train_ep=MAX_TRAIN_EP,
                    max_test_ep=MAX_TEST_EP, 
                    eps_list=eps_list,
                    log_enable=RUNNER_LOG_ENABLE)

    runner.start()
