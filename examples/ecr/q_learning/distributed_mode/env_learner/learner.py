# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import sys
import io
import numpy as np
import torch
import random
import yaml

from datetime import datetime
from maro.distributed import dist
from maro.utils import SimpleExperiencePool, Logger, LogFormat, convert_dottable

from examples.ecr.q_learning.distributed_mode.env_learner.message_type import MsgType, MsgKey
from examples.ecr.q_learning.common.dqn import QNet, DQN
from examples.utils import log, get_proxy
from examples.ecr.q_learning.common.dashboard_ex import DashboardECR

CONFIG_PATH = os.environ.get('CONFIG') or 'config.yml'

with io.open(CONFIG_PATH, 'r') as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

LOG_FOLDER = os.path.join(os.getcwd(), 'log', f"{datetime.now().strftime('%Y%m%d')}", config.experiment_name)
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

with io.open(os.path.join(LOG_FOLDER, 'config.yml'), 'w', encoding='utf8') as out_file:
    yaml.safe_dump(raw_config, out_file)



BATCH_NUM = config.train.batch_num
BATCH_SIZE = config.train.batch_size
MIN_TRAIN_EXP_NUM = config.train.min_train_experience_num  # when experience num is less than this num, agent will not train model
DQN_LOG_ENABLE = config.log.dqn.enable
DQN_LOG_DROPOUT_P = config.log.dqn.dropout_p
QNET_LOG_ENABLE = config.log.qnet.enable
LEARNING_RATE = config.train.dqn.lr
GAMMA = config.train.dqn.gamma  # Reward decay
TAU = config.train.dqn.tau  # Soft update
TARGET_UPDATE_FREQ = config.train.dqn.target_update_frequency
TRAIN_SEED = config.train.seed
DASHBOARD_ENABLE = config.dashboard.enable
DASHBOARD_HOST = config.dashboard.influxdb.host
DASHBOARD_PORT = config.dashboard.influxdb.port
DASHBOARD_USE_UDP = config.dashboard.influxdb.use_udp
DASHBOARD_UDP_PORT = config.dashboard.influxdb.udp_port

COMPONENT = 'learner'
logger = Logger(tag=COMPONENT, format_=LogFormat.simple,
                dump_folder=LOG_FOLDER, dump_mode='w', auto_timestamp=False)
proxy = get_proxy(COMPONENT, config, logger=logger)

if DASHBOARD_ENABLE:
    dashboard = DashboardECR(config.experiment_name, LOG_FOLDER)
    dashboard.setup_connection(host = DASHBOARD_HOST, port = DASHBOARD_PORT, use_udp = DASHBOARD_USE_UDP, udp_port = DASHBOARD_UDP_PORT)


@log(logger=logger)
def on_new_experience(local_instance, proxy, msg):
    """
    Handles incoming experience from environment runner. The message must contain agent_id and experience.
    """
    # put experience into experience pool
    local_instance.experience_pool.put(category_data_batches=msg.body[MsgKey.EXPERIENCE])
    policy_net_parameters = None

    # trigger trining process if got enough experience
    if local_instance.experience_pool.size['info'] > MIN_TRAIN_EXP_NUM:
        local_instance.train(msg.body[MsgKey.EPISODE], msg.body[MsgKey.AGENT_NAME])
        policy_net_parameters = local_instance.algorithm.policy_net.state_dict()

    # send updated policy net parameters to the target environment runner
    proxy.send(peer_name=msg.src, msg_type=MsgType.UPDATED_PARAMETERS,
               msg_body={MsgKey.AGENT_ID: msg.body[MsgKey.AGENT_ID],
                         MsgKey.POLICY_NET_PARAMETERS: policy_net_parameters})


@log(logger=logger)
def on_initial_net_parameters(local_instance, proxy, msg):
    """
    Handles initial net parameters from environment runner. The message must contain policy net parameters 
    and target net parameters
    """
    local_instance.init_network(msg.body[MsgKey.POLICY_NET_PARAMETERS], msg.body[MsgKey.TARGET_NET_PARAMETERS])


@log(logger=logger)
def on_env_checkout(local_instance, proxy, msg):
    """
    Handle environment runner checkout message.
    """
    local_instance.env_checkout(msg.src)


handler_dict = {MsgType.STORE_EXPERIENCE: on_new_experience,
                MsgType.INITIAL_PARAMETERS: on_initial_net_parameters,
                MsgType.ENV_CHECKOUT: on_env_checkout}


@dist(proxy=proxy, handler_dict=handler_dict)
class Learner:
    def __init__(self):
        self.experience_pool = SimpleExperiencePool()
        self._env_number = 0
        self._batch_size = BATCH_SIZE
        self._batch_num = BATCH_NUM
        self._set_seed(TRAIN_SEED)

    def init_network(self, policy_net_parameters, target_net_parameters):
        """
        Initial the algorithm for learner by the mean of net's parameters from environments

        Args:
            policy_net_parameters: Tuple(name, input_dim, hidden_dims, output_dim, dropout_p)
            target_net_parameters: Tuple(name, input_dim, hidden_dims, output_dim, dropout_p)
        """
        if not self._env_number:
            policy_net = QNet(*policy_net_parameters,
                              log_enable=True, log_folder=LOG_FOLDER)
            target_net = QNet(*target_net_parameters,
                              log_enable=True, log_folder=LOG_FOLDER)
            target_net.load_state_dict(policy_net.state_dict())

            self.algorithm = DQN(policy_net=policy_net, target_net=target_net,
                                 gamma=GAMMA, tau=TAU, target_update_frequency=TARGET_UPDATE_FREQ, lr=LEARNING_RATE,
                                 log_enable=DQN_LOG_ENABLE, log_folder=LOG_FOLDER, log_dropout_p=DQN_LOG_DROPOUT_P)

        self._env_number += 1

    def env_checkout(self, env_id):
        """
        Receive the envrionment checkout, if all environment are exitted, stop current learner
        """
        self._env_number -= 1
        if not self._env_number:
            logger.critical("Learner exited.")
            sys.exit(1)

    def train(self, episode, agent_name):
        """
        Training Process

        Args:
            episode: int
        """
        for i in range(self._batch_num):
            # prepare experiences
            idx_list = self.experience_pool.apply_multi_samplers(
                category_samplers=[('info', [(lambda i, o: (i, o['td_error']), self._batch_size)])])['info']
            sample_dict = self.experience_pool.get(category_idx_batches=[
                ('state', idx_list),
                ('reward', idx_list),
                ('action', idx_list),
                ('next_state', idx_list),
                ('info', idx_list)
            ])

            state_batch = torch.from_numpy(
                np.array(sample_dict['state'])).view(-1, self.algorithm.policy_net.input_dim)
            action_batch = torch.from_numpy(
                np.array(sample_dict['action'])).view(-1, 1)
            reward_batch = torch.from_numpy(
                np.array(sample_dict['reward'])).view(-1, 1)
            next_state_batch = torch.from_numpy(
                np.array(sample_dict['next_state'])).view(-1, self.algorithm.policy_net.input_dim)

            loss = self.algorithm.learn(state_batch=state_batch, action_batch=action_batch,
                                        reward_batch=reward_batch, next_state_batch=next_state_batch,
                                        current_ep=episode)

            # update td-error
            for i in range(len(idx_list)):
                sample_dict['info'][i]['td_error'] = loss

            self.experience_pool.update([('info', idx_list, sample_dict['info'])])
            if DASHBOARD_ENABLE:
                dashboard.upload_loss({agent_name: loss}, episode)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


if __name__ == '__main__':
    # Learner initialization
    component_name = '_'.join([COMPONENT, '0']) if 'INDEX' not in os.environ else '_'.join(
        [COMPONENT, os.environ['INDEX']])
    learner = Learner()
    learner.launch(os.environ['GROUP'], component_name)
