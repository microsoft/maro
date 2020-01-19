# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import io
import os
import random
import numpy as np
import torch
import yaml

import maro.simulator.utils.random as sim_random
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
from maro.simulator import Env
from maro.utils import SimpleExperiencePool, Logger, LogFormat, convert_dottable

from examples.ecr.q_learning.common.agent import Agent
from examples.ecr.q_learning.common.dqn import QNet, DQN
from examples.ecr.q_learning.common.state_shaping import StateShaping
from examples.ecr.q_learning.common.action_shaping import DiscreteActionShaping
from examples.ecr.q_learning.common.dashboard_ex import DashboardECR
from maro.simulator.scenarios.ecr.common import EcrEventType


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
DASHBOARD_ENABLE = config.dashboard.enable
DASHBOARD_HOST = config.dashboard.influxdb.host
DASHBOARD_PORT = config.dashboard.influxdb.port
DASHBOARD_USE_UDP = config.dashboard.influxdb.use_udp
DASHBOARD_UDP_PORT = config.dashboard.influxdb.udp_port
NUM_THREADS = 16

if config.train.reward_shaping not in {'gf', 'tc'}:
    raise ValueError('Unsupported reward shaping. Currently supported reward shaping types: "gf", "tc"')

REWARD_SHAPING = config.train.reward_shaping

####################################################### END OF INITIAL_PARAMETERS #######################################################


class Runner:
    def __init__(self, scenario: str, topology: str, max_tick: int, max_train_ep: int, max_test_ep: int,
                 eps_list: [float], log_enable: bool = True, dashboard_enable: bool = True):

        self._set_seed(TRAIN_SEED)
        self._env = Env(scenario, topology, max_tick)
        self.dashboard = None

        if dashboard_enable:
            self.dashboard = DashboardECR(config.experiment_name, LOG_FOLDER)
            self.dashboard.setup_connection(host=DASHBOARD_HOST, port=DASHBOARD_PORT,
                                            use_udp=DASHBOARD_USE_UDP, udp_port=DASHBOARD_UDP_PORT)

        self._topology = topology
        self._port_idx2name = self._env.node_name_mapping['static']
        self._vessel_idx2name = self._env.node_name_mapping['dynamic']
        self._agent_dict = self._load_agent(self._env.agent_idx_list)
        self._max_train_ep = max_train_ep
        self._max_test_ep = max_test_ep
        self._max_tick = max_tick
        self._eps_list = eps_list
        self._log_enable = log_enable
        self._dashboard_enable = dashboard_enable

        if log_enable:
            self._logger = Logger(tag='runner', format_=LogFormat.simple,
                                  dump_folder=LOG_FOLDER, dump_mode='w', auto_timestamp=False)
            self._performance_logger = Logger(tag=f'runner.performance', format_=LogFormat.none,
                                              dump_folder=LOG_FOLDER, dump_mode='w', extension_name='csv',
                                              auto_timestamp=False)
            self._performance_logger.debug(
                f"episode,epsilon,{','.join([port_name + '_booking' for port_name in self._port_idx2name.values()])},"
                f"total_booking,{','.join([port_name + '_shortage' for port_name in self._port_idx2name.values()])},"
                f"total_shortage,total_early_discharge,total_true_early_discharge,shortage_appear_tick")

    def _load_agent(self, agent_idx_list: [int]):
        self._set_seed(QNET_SEED)
        agent_dict = {}
        state_shaping = StateShaping(env=self._env,
                                     relative_tick_list=[-1, -2, -3, -4, -5, -6, -7],
                                     port_downstream_max_number=2,
                                     port_attribute_list=['empty', 'full', 'on_shipper', 'on_consignee', 'booking',
                                                          'shortage', 'fulfillment'],
                                     vessel_attribute_list=['empty', 'full', 'remaining_space'])
        action_space = [round(i * 0.1, 1) for i in range(-10, 11)]
        action_shaping = DiscreteActionShaping(action_space=action_space)
        for agent_idx in agent_idx_list:
            experience_pool = SimpleExperiencePool()
            policy_net = QNet(name=f'{self._port_idx2name[agent_idx]}.policy', input_dim=state_shaping.dim,
                              hidden_dims=[
                                  256, 128, 64], output_dim=len(action_space), dropout_p=DROPOUT_P,
                              log_enable=QNET_LOG_ENABLE, log_folder=LOG_FOLDER)
            target_net = QNet(name=f'{self._port_idx2name[agent_idx]}.target', input_dim=state_shaping.dim,
                              hidden_dims=[
                                  256, 128, 64], output_dim=len(action_space), dropout_p=DROPOUT_P,
                              log_enable=QNET_LOG_ENABLE, log_folder=LOG_FOLDER)
            target_net.load_state_dict(policy_net.state_dict())
            dqn = DQN(policy_net=policy_net, target_net=target_net,
                      gamma=GAMMA, tau=TAU, target_update_frequency=TARGET_UPDATE_FREQ, lr=LEARNING_RATE,
                      log_enable=DQN_LOG_ENABLE, log_folder=LOG_FOLDER, log_dropout_p=DQN_LOG_DROPOUT_P,
                      dashboard_enable=DASHBOARD_ENABLE, dashboard=self.dashboard)
            agent_dict[agent_idx] = Agent(agent_name=self._port_idx2name[agent_idx],
                                          topology=self._topology,
                                          port_idx2name=self._port_idx2name,
                                          vessel_idx2name=self._vessel_idx2name,
                                          algorithm=dqn, experience_pool=experience_pool,
                                          state_shaping=state_shaping, action_shaping=action_shaping,
                                          reward_shaping=REWARD_SHAPING,
                                          batch_num=BATCH_NUM, batch_size=BATCH_SIZE,
                                          min_train_experience_num=MIN_TRAIN_EXP_NUM,
                                          agent_idx_list=agent_idx_list,
                                          log_enable=AGENT_LOG_ENABLE, log_folder=LOG_FOLDER,
                                          dashboard_enable=DASHBOARD_ENABLE, dashboard=self.dashboard)

        return agent_dict

    def start(self):
        pbar = tqdm(range(self._max_train_ep))

        for ep in pbar:
            self._set_seed(TRAIN_SEED + ep)
            pbar.set_description('train episode')
            _, decision_event, is_done = self._env.step(None)

            total_early_discharge, total_true_early_discharge, shortage_appear_tick = 0, 0, -1
            while not is_done:
                action = self._agent_dict[decision_event.port_idx].choose_action(
                    decision_event=decision_event, eps=self._eps_list[ep], current_ep=ep)
                _, decision_event, is_done = self._env.step(action)

                # Update early_discharge, true_early_discharge and shortage_appear_tick, remove in the future
                early_discharge = decision_event.early_discharge
                total_early_discharge += early_discharge
                if action.quantity <= 0:
                    total_true_early_discharge += early_discharge
                shortage = sum([self._env.snapshot_list.static_nodes[self._env.tick: port_idx: ('shortage', 0)][0]
                                for port_idx in self._env.agent_idx_list])
                if shortage_appear_tick < 0 < shortage:
                    shortage_appear_tick = self._env.tick

            if self._log_enable:
                self._print_summary(ep=ep, is_train=True)

            for agent in self._agent_dict.values():
                agent.calculate_offline_rewards(snapshot_list=self._env.snapshot_list, current_ep=ep)
                agent.store_experience()
                agent.train(current_ep=ep)

            self._env.reset()

        self._test()

    def _test(self):
        pbar = tqdm(range(self._max_test_ep))
        for ep in pbar:
            self._set_seed(TEST_SEED)
            pbar.set_description('test episode')
            _, decision_event, is_done = self._env.step(None)
            while not is_done:
                action = self._agent_dict[decision_event.port_idx].choose_action(
                    decision_event=decision_event, eps=0, current_ep=ep)
                _, decision_event, is_done = self._env.step(action)

            if self._log_enable:
                self._print_summary(ep=ep, is_train=False)

            self._env.reset()

    def _print_summary(self, ep, is_train: bool = True):
        shortage_list = self._env.snapshot_list.static_nodes[
                        self._env.tick: self._env.agent_idx_list: ('acc_shortage', 0)]
        pretty_shortage_dict = OrderedDict()
        tot_shortage = 0
        for i, shortage in enumerate(shortage_list):
            pretty_shortage_dict[self._port_idx2name[i]] = shortage
            tot_shortage += shortage
        pretty_shortage_dict['total_shortage'] = tot_shortage

        booking_list = self._env.snapshot_list.static_nodes[
                       self._env.tick: self._env.agent_idx_list: ('acc_booking', 0)]
        pretty_booking_dict = OrderedDict()
        tot_booking = 0
        for i, booking in enumerate(booking_list):
            pretty_booking_dict[self._port_idx2name[i]] = booking
            tot_booking += booking
        pretty_booking_dict['total_booking'] = tot_booking

        if is_train:
            self._performance_logger.debug(
                f"{ep},{self._eps_list[ep]},{','.join([str(value) for value in pretty_booking_dict.values()])},{','.join([str(value) for value in pretty_shortage_dict.values()])},"
                f"{total_early_discharge},{total_true_early_discharge},{shortage_appear_tick}"
            )
            self._logger.critical(
                f'{self._env.name} | train | [{ep + 1}/{self._max_train_ep}] total tick: {self._max_tick}, total booking: {pretty_booking_dict}, total shortage: {pretty_shortage_dict}')
        else:
            self._logger.critical(
                f'{self._env.name} | test | [{ep + 1}/{self._max_test_ep}] total tick: {self._max_tick}, total booking: {pretty_booking_dict}, total shortage: {pretty_shortage_dict}')

        if self._dashboard_enable:
            dashboard_ep = ep
            if not is_train:
                dashboard_ep = ep + self._max_train_ep
            self.dashboard.upload_booking(pretty_booking_dict, dashboard_ep)
            self.dashboard.upload_shortage(pretty_shortage_dict, dashboard_ep)
            if not is_train:
                if ep == self._max_test_ep - 1:
                    self.dashboard.upload_to_ranklist({'enabled': True, 'name': 'test_shortage_ranklist'}, fields={
                        'shortage': pretty_shortage_dict['total_shortage']})
            if is_train:
                pretty_epsilon_dict = OrderedDict()
                for i, _ in enumerate(self._port_idx2name):
                    pretty_epsilon_dict[self._port_idx2name[i]
                    ] = self._eps_list[ep]
                self.dashboard.upload_epsilon(
                    pretty_epsilon_dict, dashboard_ep)

            events = self._env.get_finished_events()
            from_to_executed = {}

            for event in events:
                if event.event_type == EcrEventType.DISCHARGE_FULL:
                    if event.payload.from_port_idx not in from_to_executed:
                        from_to_executed[event.payload.from_port_idx] = {}
                    if event.payload.port_idx not in from_to_executed[event.payload.from_port_idx]:
                        from_to_executed[event.payload.from_port_idx][event.payload.port_idx] = 0
                    from_to_executed[event.payload.from_port_idx][event.payload.port_idx] += event.payload.quantity
            for k, v in enumerate(from_to_executed):
                for kk, vv in enumerate(from_to_executed[v]):
                    self.dashboard.upload_laden_executed(
                        {'from': self._port_idx2name[v], 'to': self._port_idx2name[vv],
                         'quantity': from_to_executed[v][vv]}, dashboard_ep)

            from_to_planed = {}

            for event in events:
                if event.event_type == EcrEventType.ORDER:
                    if event.payload.src_port_idx not in from_to_planed:
                        from_to_planed[event.payload.src_port_idx] = {}
                    if event.payload.dest_port_idx not in from_to_planed[event.payload.src_port_idx]:
                        from_to_planed[event.payload.src_port_idx][event.payload.dest_port_idx] = 0
                    from_to_planed[event.payload.src_port_idx][event.payload.dest_port_idx] += event.payload.quantity

            for k, v in enumerate(from_to_planed):
                for kk, vv in enumerate(from_to_planed[v]):
                    self.dashboard.upload_laden_planed(
                        {'from': self._port_idx2name[v], 'to': self._port_idx2name[vv],
                         'quantity': from_to_planed[v][vv]}, dashboard_ep)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        sim_random.seed(seed)


if __name__ == '__main__':
    torch.set_num_threads(NUM_THREADS)

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
                    log_enable=RUNNER_LOG_ENABLE, dashboard_enable=DASHBOARD_ENABLE)

    runner.start()
