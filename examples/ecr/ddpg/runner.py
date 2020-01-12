from collections import OrderedDict
from datetime import datetime
import io
import os
import random

import numpy as np
import torch
from tqdm import tqdm
import yaml

import maro.simulator.utils.random as sim_random
from maro.simulator import Env
from maro.utils import SimpleExperiencePool, Logger, LogFormat, convert_dottable

from examples.ecr.ddpg.ddpg_agent import Agent
from examples.ecr.ddpg.ddpg import Actor, Critic, DDPG
from examples.ecr.common.state_shaping import StateShaping
from examples.ecr.common.action_shaping import ContinuousActionShaping

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

TARGET_UPDATE_FREQ = config.train.ddpg.target_update_frequency
CRITIC_LEARNING_RATE = config.train.ddpg.critic_lr
ACTOR_LEARNING_RATE = config.train.ddpg.actor_lr
DROPOUT = config.train.ddpg.dropout
GAMMA = config.train.ddpg.gamma  # Reward decay
TAU = config.train.ddpg.tau  # Soft update
SIGMA = config.train.exploration.sigma
THETA = config.train.exploration.theta
BATCH_NUM = config.train.batch_num
BATCH_SIZE = config.train.batch_size
MIN_TRAIN_EXP_NUM = config.train.min_train_experience_num  # when experience num is less than this num, agent will not train model
REWARD_SHAPING = config.train.reward_shaping
TRAIN_SEED = config.train.seed
TEST_SEED = config.test.seed
QNET_SEED = config.qnet.seed
RUNNER_LOG_ENABLE = config.log.runner.enable
AGENT_LOG_ENABLE = config.log.agent.enable
DDPG_LOG_ENABLE = config.log.ddpg.enable
DDPG_LOG_DROPOUT = config.log.ddpg.dropout
ACTOR_LOG_ENABLE = config.log.actor.enable
CRITIC_LOG_ENABLE = config.log.critic.enable


class Runner:
    def __init__(self, scenario: str, topology: str, max_tick: int, max_train_ep: int, max_test_ep: int, log_enable: bool = True):
        self._env = Env(scenario, topology, max_tick)
        self._topology = topology
        self._port_idx2name = self._env.node_name_mapping['static']
        self._vessel_idx2name = self._env.node_name_mapping['dynamic']
        self._agent_dict = self._load_agent(
            self._env.agent_idx_list)
        self._max_train_ep = max_train_ep
        self._max_test_ep = max_test_ep
        self._max_tick = max_tick
        self._log_enable = log_enable

        if log_enable:
            self._logger = Logger(tag='runner', format_=LogFormat.simple,
                                  dump_folder=LOG_FOLDER, dump_mode='w', auto_timestamp=False)
            self._performance_logger = Logger(tag=f'runner.performance', format_=LogFormat.none,
                                              dump_folder=LOG_FOLDER, dump_mode='w', extension_name='csv',
                                              auto_timestamp=False)
            self._performance_logger.debug(
                f"episode,epsilon,{','.join([port_name + '_booking' for port_name in self._port_idx2name.values()])},total_booking,{','.join([port_name + '_shortage' for port_name in self._port_idx2name.values()])},total_shortage")

    def _load_agent(self, agent_idx_list: [int]):
        self._set_seed(QNET_SEED)
        agent_dict = {}
        state_shaping = StateShaping(env=self._env,
                                     relative_tick_list=[-1, -2, -3, -4, -5, -6, -7],
                                     port_downstream_max_number=2,
                                     port_attribute_list=['empty', 'full', 'on_shipper', 'on_consignee', 'booking',
                                                          'shortage', 'fulfillment'],
                                     vessel_attribute_list=['empty', 'full', 'remaining_space'])
        action_shaping = ContinuousActionShaping()
        for agent_idx in agent_idx_list:
            experience_pool = SimpleExperiencePool()
            actor_policy_net = Actor(name=f'{self._port_idx2name[agent_idx]}.policy_actor', 
                                     input_dim=state_shaping.dim,
                                     hidden_dims=[256, 128, 64], 
                                     output_dim=1, 
                                     dropout_actor=DROPOUT,
                                     log_enable=ACTOR_LOG_ENABLE, 
                                     log_folder=LOG_FOLDER)
            actor_target_net = Actor(name=f'{self._port_idx2name[agent_idx]}.target_actor', 
                                     input_dim=state_shaping.dim, 
                                     hidden_dims=[256, 128, 64], 
                                     output_dim=1, 
                                     dropout_actor=DROPOUT,
                                     log_enable=ACTOR_LOG_ENABLE, 
                                     log_folder=LOG_FOLDER)
            actor_target_net.load_state_dict(actor_policy_net.state_dict())

            critic_policy_net = Critic(name=f'{self._port_idx2name[agent_idx]}.policy_critic', 
                                       input_dim=state_shaping.dim,
                                       state_input_hidden_dims = [256, 128],
                                       action_input_hidden_dims = [128, 64, 32],
                                       action_dim=1, 
                                       dropout_critic=DROPOUT,
                                       log_enable=CRITIC_LOG_ENABLE, 
                                       log_folder=LOG_FOLDER)
            critic_target_net = Critic(name=f'{self._port_idx2name[agent_idx]}.target_critic', 
                                       input_dim=state_shaping.dim,
                                       state_input_hidden_dims = [256, 128],
                                       action_input_hidden_dims = [128, 64, 32],
                                       action_dim=1, 
                                       dropout_critic=DROPOUT,
                                       log_enable=CRITIC_LOG_ENABLE,
                                       log_folder=LOG_FOLDER)
            critic_target_net.load_state_dict(critic_policy_net.state_dict())

            ddpg = DDPG(actor_policy_net=actor_policy_net, actor_target_net=actor_target_net,
                        critic_policy_net=critic_policy_net, critic_target_net=critic_target_net,
                        gamma=GAMMA, tau=TAU, target_update_frequency=TARGET_UPDATE_FREQ, critic_lr=CRITIC_LEARNING_RATE, actor_lr=ACTOR_LEARNING_RATE, sigma=SIGMA, theta=THETA,
                        log_enable=DDPG_LOG_ENABLE, log_folder=LOG_FOLDER, log_dropout_ddpg=DDPG_LOG_DROPOUT)
            
            agent_dict[agent_idx] = Agent(agent_name=self._port_idx2name[agent_idx], 
                                          topology=self._topology,
                                          port_idx2name=self._port_idx2name,
                                          vessel_idx2name=self._vessel_idx2name,
                                          algorithm=ddpg, experience_pool=experience_pool,
                                          state_shaping=state_shaping, action_shaping=action_shaping,
                                          reward_shaping=REWARD_SHAPING,
                                          batch_num=BATCH_NUM, batch_size=BATCH_SIZE,
                                          min_train_experience_num=MIN_TRAIN_EXP_NUM,
                                          agent_idx_list=self._env.agent_idx_list,
                                          log_enable=AGENT_LOG_ENABLE, log_folder=LOG_FOLDER)

        return agent_dict

    def start(self):
        pbar = tqdm(range(self._max_train_ep))
        for ep in pbar:
            self._set_seed(TRAIN_SEED + ep)
            pbar.set_description('train episode')
            _, decision_event, is_done = self._env.step(None)

            while not is_done:
                action = self._agent_dict[decision_event.port_idx].choose_action(
                    decision_event=decision_event, is_test=False, current_ep=ep)
                _, decision_event, is_done = self._env.step(action)

            need_train = True
            for agent in self._agent_dict.values():
                agent.fulfill_cache(agent_idx_list=self._env.agent_idx_list,
                                    snapshot_list=self._env.snapshot_list,
                                    current_ep=ep)
                agent.put_experience()
                agent.clear_cache()
                if agent.experience_pool.size['info'] < MIN_TRAIN_EXP_NUM:
                    need_train = False

            if need_train:
                for agent in self._agent_dict.values():
                    agent.train(current_ep=ep)
                    # agent.experience_pool.reset()

            self._print_summary(ep=ep, is_train=True)

            self._env.reset()

            # self._set_seed(TRAIN_SEED + ep)
            # pbar.set_description('train episode')
            # _, decision_event, is_done = self._env.step(None)

            # while not is_done:
            #     action = self._agent_dict[decision_event.port_idx].choose_action(
            #         decision_event=decision_event, is_test = True, current_ep=ep)
            #     _, decision_event, is_done = self._env.step(action)

            # self._print_summary(ep=ep, is_train=False)

            # self._env.reset()

        self._test()

    def _test(self):
        pbar = tqdm(range(self._max_test_ep))
        for ep in pbar:
            self._set_seed(TEST_SEED)
            pbar.set_description('test episode')
            _, decision_event, is_done = self._env.step(None)
            while not is_done:
                action = self._agent_dict[decision_event.port_idx].choose_action(
                    decision_event=decision_event, is_test=True, current_ep=ep)
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
                f"{ep},{','.join([str(value) for value in pretty_booking_dict.values()])},{','.join([str(value) for value in pretty_shortage_dict.values()])}")
            self._logger.critical(
                f'{self._env.name} | train | reward shaping: {REWARD_SHAPING} | [{ep + 1}/{self._max_train_ep}] total tick: {self._max_tick}, \n total booking: {pretty_booking_dict}, \n total shortage: {pretty_shortage_dict}')
        else:
            self._logger.critical(
                f'{self._env.name} | test | [{ep + 1}/{self._max_test_ep}] total tick: {self._max_tick}, total booking: {pretty_booking_dict}, total shortage: {pretty_shortage_dict}')

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        sim_random.seed(seed)



if __name__ == '__main__':
    runner = Runner(scenario=SCENARIO, topology=TOPOLOGY,
                    max_tick=MAX_TICK, max_train_ep=MAX_TRAIN_EP,
                    max_test_ep=MAX_TEST_EP,
                    log_enable=RUNNER_LOG_ENABLE)

    runner.start()
