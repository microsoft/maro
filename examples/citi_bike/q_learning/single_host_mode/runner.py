# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import io
import os
import random
import numpy as np
import pandas as pd
import torch
import yaml
import time

import maro.simulator.utils.random as sim_random
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
from maro.simulator import Env
from maro.utils import SimpleExperiencePool, Logger, LogFormat, convert_dottable

from maro.simulator.scenarios.bike.common import Action
from examples.citi_bike.q_learning.common.agent import Agent
from examples.citi_bike.q_learning.common.dqn import QNet, DQN
from examples.citi_bike.q_learning.common.reward_shaping import TruncateReward
from examples.citi_bike.q_learning.common.state_shaping import StateShaping
from examples.citi_bike.q_learning.common.action_shaping import DiscreteActionShaping
from examples.citi_bike.q_learning.common.citi_bike_dashboard import Dashboardciti_bike, RanklistColumns
from maro.simulator.scenarios.bike.business_engine import BikeEventType


####################################################### START OF INITIAL_PARAMETERS #######################################################

CONFIG_PATH = os.environ.get('CONFIG') or '/home/xinran/maro/examples/citi_bike/q_learning/single_host_mode/config.yml'
#/home/xinran/maro/examples/citi_bike/q_learning/single_host_mode/

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
TEST_TOPOLOGY = config.test.topology
MAX_TICK = config.env.max_tick
TEST_TICK = config.test.max_tick
MAX_TRAIN_EP = config.train.max_ep
MAX_TEST_EP = config.test.max_ep
MAX_EPS = config.train.exploration.max_eps
PHASE_SPLIT_POINT = config.train.exploration.phase_split_point  # exploration two phase split point
FIRST_PHASE_REDUCE_PROPORTION = config.train.exploration.first_phase_reduce_proportion  # first phase reduce prostationion of max_eps
TARGET_UPDATE_FREQ = config.train.dqn.target_update_frequency
LEARNING_RATE = config.train.dqn.lr
DROPOUT_P = config.train.dqn.dropout_p
GAMMA = config.train.dqn.gamma  # Reward decay
TAU = config.train.dqn.tau  # Soft update
BATCH_NUM = config.train.batch_num
BATCH_SIZE = config.train.batch_size
MIN_TRAIN_EXP_NUM = config.train.min_train_experience_num  # when experience num is less than this num, agent will not train model
COST_FACTOR = config.train.reward.cost_factor
SHORTAGE_FACTOR = config.train.reward.shortage_factor
TRAIN_SEED = config.train.seed
TEST_SEED = config.test.seed
QNET_SEED = config.qnet.seed
RUNNER_LOG_ENABLE = config.log.runner.enable
AGENT_LOG_ENABLE = config.log.agent.enable
DQN_LOG_ENABLE = config.log.dqn.enable
DQN_LOG_DROPOUT_P = config.log.dqn.dropout_p
QNET_LOG_ENABLE = config.log.qnet.enable

if config.train.reward_shaping not in {'gf', 'tc'}:
    raise ValueError('Unsupstationed reward shaping. Currently supstationed reward shaping types: "gf", "tc"')

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
        self._env = Env(scenario, topology, max_tick, tick_units=60)
        self._test_env = Env(scenario, TEST_TOPOLOGY, TEST_TICK)
        # self._station_idx2name = self._env.node_name_mapping
        self._station_idx2name = {key:key for key in self._env.agent_idx_list}
        self._agent_dict = self._load_agent(self._env.agent_idx_list)
        self._station_name2idx = {}
        for idx in self._station_idx2name.keys():
            self._station_name2idx[self._station_idx2name[idx]] = idx
        self._time_cost = OrderedDict()
        if log_enable:
            self._logger = Logger(tag='runner', format_=LogFormat.simple,
                                  dump_folder=LOG_FOLDER, dump_mode='w', auto_timestamp=False)
        #     self._performance_logger = Logger(tag=f'runner.performance', format_=LogFormat.none,
        #                                       dump_folder=LOG_FOLDER, dump_mode='w', extension_name='csv',
        #                                       auto_timestamp=False)

    def _load_agent(self, agent_idx_list: [int]):
        if DASHBOARD_ENABLE:
            self._dashboard = Dashboardciti_bike(EXPERIMENT_NAME, LOG_FOLDER if DASHBOARD_LOG_ENABLE else None,
                                            host=DASHBOARD_HOST,
                                            port=DASHBOARD_PORT,
                                            use_udp=DASHBOARD_USE_UDP,
                                            udp_port=DASHBOARD_UDP_PORT)
            self._ranklist_enable = RANKLIST_ENABLE
            self._dashboard.update_ranklist_info(
                info={
                    RanklistColumns.experiment.value: self._dashboard.experiment,
                    RanklistColumns.author.value: AUTHOR,
                    RanklistColumns.commit.value: COMMIT,
                    RanklistColumns.initial_lr.value: LEARNING_RATE,
                    RanklistColumns.train_ep.value: self._max_train_ep,
                    'scenario': self._scenario,
                    'topology': self._topology,
                    'max_tick': self._max_tick
                })
            self._dashboard.update_static_info(
                info={
                'scenario': self._scenario,
                'topology': self._topology,
                'max_tick': self._max_tick,
                'max_train_ep': self._max_train_ep,
                'max_test_ep': self._max_test_ep,
                'author': AUTHOR,
                'commit': COMMIT,
                'initial_lr': LEARNING_RATE
            })

        else:
            self._dashboard = None
        self._set_seed(QNET_SEED)
        agent_dict = {}
        state_shaping = StateShaping(env=self._env,
                                     relative_tick_list=[-1, -2, -3, -4, -5, -6, -7],
                                     neighbor_number = 6,
                                     station_attribute_list=["bikes","fullfillment","trip_requirement","shortage","capacity",
                                     "unknow_gendors","males","females","weekday","weather","holiday","temperature","subscriptor","customer","extra_cost"])
        action_space = [round((1/6)*i,2) for i in range(0, 6)]
        action_shaping = DiscreteActionShaping(action_space=action_space)
        if REWARD_SHAPING == 'tc':
            reward_shaping = TruncateReward(env=self._env, agent_idx_list=agent_idx_list, log_folder=LOG_FOLDER, 
                                        cost_factor= COST_FACTOR, shortage_factor = SHORTAGE_FACTOR)
        else:
            raise ValueError('Unsuported Reward Shaping')

        for agent_idx in agent_idx_list:
            experience_pool = SimpleExperiencePool()
            policy_net = QNet(name=f'{self._station_idx2name[agent_idx]}.policy', input_dim=state_shaping.dim,
                              hidden_dims=[
                                  256, 128, 64], output_dim=len(action_space), dropout_p=DROPOUT_P,
                              log_folder=LOG_FOLDER if QNET_LOG_ENABLE else None)
            target_net = QNet(name=f'{self._station_idx2name[agent_idx]}.target', input_dim=state_shaping.dim,
                              hidden_dims=[
                                  256, 128, 64], output_dim=len(action_space), dropout_p=DROPOUT_P,
                              log_folder=LOG_FOLDER if QNET_LOG_ENABLE else None)
            target_net.load_state_dict(policy_net.state_dict())
            dqn = DQN(policy_net=policy_net, target_net=target_net,
                      gamma=GAMMA, tau=TAU, target_update_frequency=TARGET_UPDATE_FREQ, lr=LEARNING_RATE,
                      log_folder=LOG_FOLDER if DQN_LOG_ENABLE else None, log_dropout_p=DQN_LOG_DROPOUT_P,
                      dashboard=self._dashboard)
            agent_dict[agent_idx] = Agent(agent_name=self._station_idx2name[agent_idx],
                                          algorithm=dqn, experience_pool=experience_pool,
                                          state_shaping=state_shaping,
                                          action_shaping=action_shaping,
                                          reward_shaping=reward_shaping,
                                          batch_num=BATCH_NUM, batch_size=BATCH_SIZE,
                                          min_train_experience_num=MIN_TRAIN_EXP_NUM,
                                          log_folder=LOG_FOLDER if AGENT_LOG_ENABLE else None,
                                          dashboard=self._dashboard)

        return agent_dict

    def start(self):
        pbar = tqdm(range(self._max_train_ep))

        for ep in pbar:
            time_dict = OrderedDict()
            ep_start = time.time()
            if self._dashboard is not None:
                # set testing progress
                if ep == 0:
                    self._dashboard.update_dynamic_info(info={'is_train': False, 'current_ep': 0, 'ep_progress': f'{0}/{self._max_test_ep}'})
                # set training progress
                self._dashboard.update_dynamic_info(info={'is_train': True, 'current_ep': ep, 'ep_progress': f'{ep+1}/{self._max_train_ep}'})
            self._set_seed(TRAIN_SEED + ep)
            pbar.set_description('train episode')
            env_start = time.time()
            _, decision_event, is_done =self._env.step(None)
            feature_list = []#[0]*3*len(self._env.agent_idx_list)
            while not is_done:
                action = self._agent_dict[decision_event.cell_idx].choose_action(
                    decision_event=decision_event, eps=self._eps_list[ep], current_ep=ep, snapshot_list= self._env.snapshot_list)
                _, decision_event, is_done = self._env.step(action)
                #feature_list += self._env.snapshot_list.static_nodes[
                #        self._env.tick: self._env.agent_idx_list: (['shortage','trip_requirement','extra_cost'], 0)]
            feature_list.append(self._env.snapshot_list.static_nodes[:self._env.agent_idx_list: ('shortage', 0)].reshape(-1,len(self._env.agent_idx_list)).sum(0))
            feature_list.append(self._env.snapshot_list.static_nodes[:self._env.agent_idx_list: ('trip_requirement', 0)].reshape(-1,len(self._env.agent_idx_list)).sum(0))
            feature_list.append(self._env.snapshot_list.static_nodes[:self._env.agent_idx_list: ('extra_cost', 0)].reshape(-1,len(self._env.agent_idx_list)).sum(0))
            time_dict['env_time'] = time.time() - env_start
            time_dict['train_time'] = 0
            for agent in self._agent_dict.values():
                train_start = time.time()
                agent.store_experience(current_ep=ep)
                train_start = time.time()
                agent.train(current_ep=ep)
                time_dict['train_time'] += time.time() - train_start

            if self._log_enable:
                self._print_summary(ep=ep, feature_list= feature_list, mode = 'train')

            self._env.reset()

            time_dict['ep_time'] = time.time() - ep_start
            time_dict['other_time'] = time_dict['ep_time'] - time_dict['train_time'] - time_dict['env_time']
            self._time_cost[ep] = time_dict

            if self._dashboard is not None:
                self._dashboard.upload_exp_data(fields=self._time_cost[ep], ep=ep, tick=None, measurement='bike_time_cost')
        self._test()
        self._baseline()

    def _test(self):
        pbar = tqdm(range(self._max_test_ep))
        for ep in pbar:
            time_dict = OrderedDict()
            ep_start = time.time()
            if self._dashboard is not None:
                self._dashboard.update_dynamic_info(info={'is_train': False, 'current_ep': ep, 'ep_progress': f'{ep+1}/{self._max_test_ep}'})
            self._set_seed(TEST_SEED + ep)
            pbar.set_description('test episode')
            env_start = time.time()
            _, decision_event, is_done = self._test_env.step(None)
            feature_list = [] # [0]*3*len(self._test_env.agent_idx_list)
            while not is_done:
                action = self._agent_dict[decision_event.cell_idx].choose_action(
                    decision_event=decision_event, eps=0, current_ep=ep, snapshot_list= self._test_env.snapshot_list)
                _, decision_event, is_done = self._test_env.step(action)
                # feature_list += self._test_env.snapshot_list.static_nodes[
                #         self._test_env.tick: self._test_env.agent_idx_list: (['shortage','trip_requirement','extra_cost'], 0)]
            feature_list.append(self._env.snapshot_list.static_nodes[:self._env.agent_idx_list: ('shortage', 0)].reshape(-1,len(self._env.agent_idx_list)).sum(0))
            feature_list.append(self._env.snapshot_list.static_nodes[:self._env.agent_idx_list: ('trip_requirement', 0)].reshape(-1,len(self._env.agent_idx_list)).sum(0))
            feature_list.append(self._env.snapshot_list.static_nodes[:self._env.agent_idx_list: ('extra_cost', 0)].reshape(-1,len(self._env.agent_idx_list)).sum(0))
            time_dict['env_time'] = time.time() - env_start
            if self._log_enable:
                self._print_summary(ep=ep, feature_list= feature_list, mode='test')

            self._test_env.reset()

            time_dict['ep_time'] = time.time() - ep_start
            time_dict['other_time'] = time_dict['ep_time'] - time_dict['env_time']
            self._time_cost[ep + self._max_train_ep] = time_dict

            if self._dashboard is not None:
                self._dashboard.upload_exp_data(fields=time_dict, ep=ep + self._max_train_ep, tick=None, measurement='bike_time_cost')

    def _baseline(self):
        pbar = tqdm(range(self._max_test_ep))
        for ep in pbar:
            time_dict = OrderedDict()
            ep_start = time.time()
            self._set_seed(4096 + ep)
            pbar.set_description('baseline no action episode')
            env_start = time.time()
            _, decision_event, is_done =self._env.step(None)
            feature_list = [] #*3*len(self._env.agent_idx_list)
            while not is_done:
                _, decision_event, is_done =self._env.step(Action(0,1,0))
                # feature_list += self._env.snapshot_list.static_nodes[
                #         self._env.tick: self._env.agent_idx_list: (['shortage','trip_requirement','extra_cost'], 0)]
            feature_list.append(self._env.snapshot_list.static_nodes[:self._env.agent_idx_list: ('shortage', 0)].reshape(-1,len(self._env.agent_idx_list)).sum(0))
            feature_list.append(self._env.snapshot_list.static_nodes[:self._env.agent_idx_list: ('trip_requirement', 0)].reshape(-1,len(self._env.agent_idx_list)).sum(0))
            feature_list.append(self._env.snapshot_list.static_nodes[:self._env.agent_idx_list: ('extra_cost', 0)].reshape(-1,len(self._env.agent_idx_list)).sum(0))
            
            if self._log_enable:
                self._print_summary(ep=ep, feature_list= feature_list, mode='no_action')
            

            self._env.reset()
        
        pbar = tqdm(range(self._max_test_ep))
        for ep in pbar:
            time_dict = OrderedDict()
            ep_start = time.time()
            self._set_seed(8192 + ep)
            pbar.set_description('baseline random action episode')
            env_start = time.time()
            _, decision_event, is_done =self._env.step(None)
            feature_list = []#[0]*3*len(self._env.agent_idx_list)
            while not is_done:
                action = self._agent_dict[decision_event.cell_idx].choose_action(
                    decision_event=decision_event, eps=1, current_ep=ep, snapshot_list= self._env.snapshot_list)
                _, decision_event, is_done =self._env.step(action)
                #feature_list += self._env.snapshot_list.static_nodes[
                #        self._env.tick: self._env.agent_idx_list: (['shortage','trip_requirement','extra_cost'], 0)]
            feature_list.append(self._env.snapshot_list.static_nodes[:self._env.agent_idx_list: ('shortage', 0)].reshape(-1,len(self._env.agent_idx_list)).sum(0))
            feature_list.append(self._env.snapshot_list.static_nodes[:self._env.agent_idx_list: ('trip_requirement', 0)].reshape(-1,len(self._env.agent_idx_list)).sum(0))
            feature_list.append(self._env.snapshot_list.static_nodes[:self._env.agent_idx_list: ('extra_cost', 0)].reshape(-1,len(self._env.agent_idx_list)).sum(0))
            if self._log_enable:
                self._print_summary(ep=ep, feature_list= feature_list, mode='random_action')

            self._env.reset()

    def _print_summary(self, ep, feature_list, mode = 'train'):
        # feature_list = np.array(feature_list).reshape(-1,3).transpose((1,0))
        #  shortage_list, requirement_list, cost_list
        
        pretty_shortage_dict = OrderedDict()
        tot_shortage = 0
        for i, shortage in enumerate(feature_list[0]):
            pretty_shortage_dict[str(self._station_idx2name[i])] = shortage
            tot_shortage += shortage
        pretty_shortage_dict['total'] = tot_shortage

        pretty_requirement_dict = OrderedDict()
        trip_requirement = 0
        for i, requirement in enumerate(feature_list[1]):
            pretty_requirement_dict[str(self._station_idx2name[i])] = requirement
            trip_requirement += requirement
        pretty_requirement_dict['total'] = trip_requirement

        pretty_cost_dict = OrderedDict()
        tot_cost = 0
        for i, cost in enumerate(feature_list[2]):
            pretty_cost_dict[str(self._station_idx2name[i])] = cost
            tot_cost += cost
        pretty_cost_dict['total'] = tot_cost

        if mode == 'train':
            # self._performance_logger.debug(
            #     f"{ep},{self._eps_list[ep]},{','.join([str(value) for value in pretty_booking_dict.values()])},{','.join([str(value) for value in pretty_shortage_dict.values()])}")
            self._logger.critical(
                f'{self._env.name} | train | [{ep + 1}/{self._max_train_ep}] total tick: {self._max_tick}, fullfillment ratio: {round((trip_requirement-tot_shortage)/trip_requirement,2)}, exta cost: {tot_cost}, trip requirement: {trip_requirement}, total shortage: {tot_shortage}')
        else:
            self._logger.critical(
                f'{self._env.name} | {mode} | [{ep + 1}/{self._max_test_ep}] total tick: {self._max_tick}, fullfillment ratio: {round((trip_requirement-tot_shortage)/trip_requirement,2)}, exta cost: {tot_cost}, trip requirement: {trip_requirement}, total shortage: {tot_shortage}')

        if self._dashboard is not None and mode != 'no_action':
            self._send_to_dashboard(ep, pretty_shortage_dict, pretty_requirement_dict, mode)

    def _send_to_dashboard(self,
                           ep,
                           pretty_shortage_dict,
                           pretty_booking_dict,
                           mode: str = "train"):
        # Test ep follows Train ep
        dashboard_ep = ep
        if mode == "train":
            env = self._env
        elif mode == "test":
            dashboard_ep = ep + self._max_train_ep
            env = self._test_env
        else:
            return

        # Upload data for experiment info

        if dashboard_ep == 0:
            self._dashboard.upload_exp_data(fields=Dashboardciti_bike.static_info, ep=None, tick=None, measurement='bike_static_info')
        self._dashboard.upload_exp_data(fields=Dashboardciti_bike.dynamic_info, ep=dashboard_ep, tick=None, measurement='bike_dynamic_info')

        # Upload data for ep shortage and ep booking
        self._dashboard.upload_exp_data(fields=pretty_booking_dict, ep=dashboard_ep, tick=None, measurement='bike_booking')
        self._dashboard.upload_exp_data(fields=pretty_shortage_dict, ep=dashboard_ep, tick=None, measurement='bike_shortage')

        pretty_fulfill_dict = OrderedDict()
        for i in range(len(self._station_idx2name)):
            if pretty_booking_dict[str(self._station_idx2name[i])] > 0:
                pretty_fulfill_dict[str(self._station_idx2name[i])] = (pretty_booking_dict[str(self._station_idx2name[i])] - pretty_shortage_dict[str(self._station_idx2name[i])]) / pretty_booking_dict[str(self._station_idx2name[i])] * 100
        if pretty_booking_dict['total'] > 0:
            pretty_fulfill_dict['total_fulfill'] = (pretty_booking_dict['total'] - pretty_shortage_dict['total']) / pretty_booking_dict['total'] * 100
        self._dashboard.upload_exp_data(fields=pretty_fulfill_dict, ep=dashboard_ep, tick=None, measurement='bike_fulfill')

        # Pick and upload data for rank list
        if mode == 'test':
            if ep == self._max_test_ep - 1 and self._ranklist_enable:
                model_size = 0
                experience_qty = 0
                for agent in self._agent_dict.values():
                    model_size += agent.model_size
                    experience_qty += agent.experience_quantity
                self._dashboard.upload_to_ranklist(
                    ranklist='experiment_ranklist',
                    fields={
                        RanklistColumns.shortage.value:
                        pretty_shortage_dict['total'],
                        RanklistColumns.experience_quantity.value:
                        experience_qty,
                        RanklistColumns.model_size.value:
                        model_size,
                    })

        # Pick and upload data for epsilon
        if mode == "train":
            pretty_epsilon_dict = OrderedDict()
            for i, _ in enumerate(self._station_idx2name):
                pretty_epsilon_dict[str(self._station_idx2name[i])] = self._eps_list[ep]
            self._dashboard.upload_exp_data(fields=pretty_epsilon_dict, ep=dashboard_ep, tick=None, measurement='bike_epsilon')

        # Prepare usage and delayed laden data cache
        usage_list = env.snapshot_list.static_nodes[::(['bikes'], 0)]
        pretty_usage_list = usage_list.reshape(self._max_tick, len(self._station_idx2name) )
        capacity_list = env.snapshot_list.static_nodes[0::(['capacity'], 0)]
        pretty_capacity_df = capacity_list.reshape(1, len(self._station_idx2name) )

        # TODO: remove after confirmed no longer needed
        # delayed_laden_list = self._env.snapshot_list.matrix[[x for x in range(0, self._max_tick)]:"full_on_stations"]
        # pretty_delayed_laden_list = delayed_laden_list.reshape(self._max_tick, len(self._station_idx2name), len(self._station_idx2name))

        from_to_executed = {}
        from_to_planed = {}
        # TODO: remove after confirmed no longer needed
        # pretty_early_discharge_dict = {}
        # pretty_delayed_laden_dict = {}

        pretty_tml_cost_dict = {}

        # Check events and pick data for usage, delayed laden, laden planed, laden executed, early discharge, actual_action, tml cost
        events = env.get_finished_events()
        for event in events:
            # Pick data for ep laden executed
            if event.event_type == BikeEventType.BikeReturn:
                if event.payload.from_cell not in from_to_executed:
                    from_to_executed[event.payload.from_cell] = {}
                if event.payload.to_cell not in from_to_executed[event.payload.from_cell]:
                    from_to_executed[event.payload.from_cell][event.payload.to_cell] = 0
                from_to_executed[event.payload.from_cell][event.payload.to_cell] += event.payload.number
            # Pick data for ep laden planed
            elif event.event_type == BikeEventType.TripRequirement:
                if event.payload.from_cell not in from_to_planed:
                    from_to_planed[event.payload.from_cell] = {}
                if event.payload.to_cell not in from_to_planed[event.payload.from_cell]:
                    from_to_planed[event.payload.from_cell][event.payload.to_cell] = 0
                from_to_planed[event.payload.from_cell][event.payload.to_cell] += event.payload.number
            # TODO: bring back after have decision
            # Pick and upload data for event early discharge, actual_action, tml cost
            # elif event.event_type == BikeEventType.PENDING_DECISION:
            #     station_name = self._station_idx2name[event.payload.station_idx]
            #     pretty_early_discharge_dict[station_name] = pretty_early_discharge_dict.get(station_name, 0) + event.payload.early_discharge
            #     self._dashboard.upload_exp_data(fields={station_name: event.payload.early_discharge}, ep=dashboard_ep, tick=event.tick, measurement='event_early_discharge')
            #     event_tml_cost = event.payload.early_discharge
            #     for action in event.immediate_event_list:
            #         for action_payload in action.payload:
            #             event_tml_cost += abs(action_payload.quantity)
            #         vessel_name = self._vessel_idx2name[action_payload.vessel_idx]
            #         route_name = self._env.configs['vessels'][vessel_name]['route']['route_name']
            #         self._dashboard.upload_exp_data(fields={f'actual_action_of_{station_name}_on_{route_name}':action_payload.quantity}, ep=dashboard_ep, tick=event.tick, measurement='actual_action')
            #     pretty_tml_cost_dict[station_name] = pretty_tml_cost_dict.get(station_name, 0) + event_tml_cost
            #     self._dashboard.upload_exp_data(fields={station_name: event_tml_cost}, ep=dashboard_ep, tick=event.tick, measurement='event_tml_cost')

            elif event.event_type == BikeEventType.BikeReceived:
                cur_tick = event.tick
            
            # decison event from which upload actual action and tml cost
            elif event.event_type == 0:
                cur_tick = event.tick
                cell_idx = event.payload.cell_idx
                cell_name = str(self._station_idx2name[cell_idx])
                event_tml_cost = 0
                for action_event in event.immediate_event_list:
                    for action in action_event.payload:
                        action_num = action.number
                        action_target = action.to_cell
                        target_name = str(self._station_idx2name[action_target])
                        self._dashboard.upload_exp_data(fields={f'actual_action_of_{cell_name}_to_{target_name}':action_num}, ep=dashboard_ep, tick=cur_tick, measurement='bike_actual_action')
                        event_tml_cost += action.number
                pretty_tml_cost_dict[cell_name] = pretty_tml_cost_dict.get(cell_name, 0) + event_tml_cost
                self._dashboard.upload_exp_data(fields={cell_name: event_tml_cost}, ep=dashboard_ep, tick=event.tick, measurement='bike_event_tml_cost')


        # Upload data for ep laden_planed and ep laden_executed
        for laden_source in from_to_executed.keys():
            for laden_dest in from_to_executed[laden_source].keys():
                self._dashboard.upload_exp_data(
                    fields={
                        'from': str(self._station_idx2name[laden_source]),
                        'to': str(self._station_idx2name[laden_dest]),
                        'quantity': from_to_executed[laden_source][laden_dest]
                    }, 
                    ep=dashboard_ep, tick=None, measurement='bike_ride_executed')

        for laden_source in from_to_planed.keys():
            for laden_dest in from_to_planed[laden_source].keys():
                self._dashboard.upload_exp_data(
                    fields={
                        'from': str(self._station_idx2name[laden_source]),
                        'to': str(self._station_idx2name[laden_dest]),
                        'quantity': from_to_planed[laden_source][laden_dest]
                    }, 
                    ep=dashboard_ep, tick=None, measurement='bike_ride_planed')

        # Upload data for ep tml cost
        total_tml_cost = 0
        for tml_cost in pretty_tml_cost_dict.values():
            total_tml_cost += tml_cost
        pretty_tml_cost_dict['total'] = total_tml_cost

        self._dashboard.upload_exp_data(fields=pretty_tml_cost_dict, ep=dashboard_ep, tick=None, measurement='bike_tml_cost')

        # Pick and upload data for event shortage
        ep_shortage_list = env.snapshot_list.static_nodes[:env.agent_idx_list:('shortage',0)]
        pretty_ep_shortage_list = ep_shortage_list.reshape(self._max_tick, len(self._station_idx2name))
        for i in range(self._max_tick):
            need_upload = False
            pretty_ep_shortage_dict = OrderedDict()
            for j in range(len(self._station_idx2name)):
                pretty_ep_shortage_dict[str(self._station_idx2name[j])] = pretty_ep_shortage_list[i][j]
                if pretty_ep_shortage_list[i][j] > 0:
                    need_upload = True
            if need_upload:
                self._dashboard.upload_exp_data(fields=pretty_ep_shortage_dict, ep=dashboard_ep, tick=i, measurement='bike_event_shortage')

        # Pick and upload data for event vessel usage
        pretty_usage_df = pd.DataFrame(pretty_usage_list)
        shift_usage_df = pretty_usage_df.shift(periods=1)
        result_df = (pretty_usage_df != shift_usage_df)
        result_df.apply(lambda x: self._send_usage(x , pretty_usage_df, pretty_capacity_df, dashboard_ep), axis=1)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        sim_random.seed(seed)

    def _send_usage(self, series_x, usage, capacity, ep):
        tick = series_x.name
        for idx in series_x.index:
            if series_x[idx]:
                cur_usage = {
                    'station': str(idx),
                    'bikes': usage[idx][series_x.name],
                    'docks': capacity[0][idx] - usage[idx][series_x.name]
                }
                self._dashboard.upload_exp_data(fields=cur_usage, ep=ep, tick=tick, measurement='bike_station_usage')


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

    runner = Runner(scenario=SCENARIO, topology=TOPOLOGY,
                    max_tick=MAX_TICK, max_train_ep=MAX_TRAIN_EP,
                    max_test_ep=MAX_TEST_EP, eps_list=eps_list,
                    log_enable=RUNNER_LOG_ENABLE)

    runner.start()
    # runner._baseline()
