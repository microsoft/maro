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
from maro.utils import SimpleExperiencePool, Logger, LogFormat, convert_dottable

from maro.distributed import dist, Proxy, Message
from examples.ecr.q_learning.distributed_mode.message_type import MsgType, PayloadKey
from examples.ecr.q_learning.common.dqn import QNet, DQN
from examples.utils import log, get_peers
from examples.ecr.q_learning.common.ecr_dashboard import DashboardECR


CONFIG_PATH = os.environ.get('CONFIG') or 'config.yml'

with io.open(CONFIG_PATH, 'r') as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

LOG_FOLDER = os.path.join(os.getcwd(), 'log', f"{datetime.now().strftime('%Y%m%d')}", config.experiment_name)
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

with io.open(os.path.join(LOG_FOLDER, 'config.yml'), 'w', encoding='utf8') as out_file:
    yaml.safe_dump(raw_config, out_file)

ANY_SOURCE = True
ANY_TYPE = True

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
TRAIN_MODE=config.distributed.mode
DASHBOARD_ENABLE = config.dashboard.enable
DASHBOARD_LOG_ENABLE = config.log.dashboard.enable
DASHBOARD_HOST = config.dashboard.influxdb.host
DASHBOARD_PORT = config.dashboard.influxdb.port
DASHBOARD_USE_UDP = config.dashboard.influxdb.use_udp
DASHBOARD_UDP_PORT = config.dashboard.influxdb.udp_port

COMPONENT_TYPE = os.environ['COMPTYPE']
COMPONENT_ID = os.environ['COMPID']
COMPONENT_NAME = '.'.join([COMPONENT_TYPE, COMPONENT_ID])
logger = Logger(tag=COMPONENT_NAME, format_=LogFormat.simple,
                dump_folder=LOG_FOLDER, dump_mode='w', auto_timestamp=False)

proxy = Proxy(group_name=os.environ['GROUP'],
              component_name=COMPONENT_NAME,
              peer_name_list=get_peers(COMPONENT_TYPE, config.distributed),
              redis_address=(config.redis.host, config.redis.port),
              logger=logger, msg_request=msg_request)

pending_envs = set(proxy.peers)  # environments the learner expects experiences from, required for forced sync

if DASHBOARD_ENABLE:
    dashboard = DashboardECR(config.experiment_name, LOG_FOLDER, host=DASHBOARD_HOST,
                             port=DASHBOARD_PORT, use_udp=DASHBOARD_USE_UDP, udp_port=DASHBOARD_UDP_PORT)


@log(logger=logger)
def on_new_experience(local_instance, proxy, message):
    """
    Handles incoming experience from environment runner. The message must contain agent_id and experience.
    """
    # put experience into experience pool
    peer_list = []
    for msg in message:
        episode = msg.payload[PayloadKey.EPISODE]
        agent_name = msg.payload[PayloadKey.AGENT_NAME]
        peer_list.append(msg.source)
        exp = msg.payload[PayloadKey.EXPERIENCE]
        local_instance.experience_pool.put(category_data_batches=[(name, cache) for name, cache in exp.items()])

    if local_instance.experience_pool.size['info'] > MIN_TRAIN_EXP_NUM:
        local_instance.train(episode, agent_name)
        policy_net_parameters = local_instance.algorithm.policy_net.state_dict()
        # send updated policy net parameters to the target environment runner
        for env in peer_list:
            proxy.send(Message(type=MsgType.UPDATED_PARAMETERS, source=proxy.name, destination=env,
                                payload={PayloadKey.AGENT_ID: message.payload[PayloadKey.AGENT_ID],
                                        PayloadKey.POLICY_NET_PARAMETERS: policy_net_parameters}))
    else:
        for env in peer_list:
            proxy.send(Message(type=MsgType.NO_UPDATED_PARAMETERS, source=proxy.name, destination=env))


@log(logger=logger)
def on_initial_net_parameters(local_instance, proxy, message):
    """
    Handles initial net parameters from environment runner. The message must contain policy net parameters
    and target net parameters
    """
    local_instance.init_network(message.payload[PayloadKey.POLICY_NET_PARAMETERS], message.payload[PayloadKey.TARGET_NET_PARAMETERS])


@log(logger=logger)
def on_env_checkout(local_instance, proxy, message):
    """
    Handle environment runner checkout message. If all environments have checked out, stop current learner process
    """
    if message.source in pending_envs:
        pending_envs.remove(message.source)
        if len(pending_envs) == 0:
            logger.critical(f"{COMPONENT_NAME} exited")
            sys.exit(0)

handler_dict = {MsgType.STORE_EXPERIENCE: on_new_experience,
                MsgType.INITIAL_PARAMETERS: on_initial_net_parameters,
                MsgType.ENV_CHECKOUT: on_env_checkout}

# TODO: add component support; support float
handler_dict = [{'constraint': {(ANY_SOURCE, MsgType): num, 
                            (env, ANY_TYPE): num}
                 'handler_fn': on_new_experience},
                {'request': {(env, MsgType): num, 
                            (env, MsgType): num},
                 'handler_fn': on_new_experience}]


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
                              log_folder=LOG_FOLDER)
            target_net = QNet(*target_net_parameters,
                              log_folder=LOG_FOLDER)
            target_net.load_state_dict(policy_net.state_dict())

            self.algorithm = DQN(policy_net=policy_net, target_net=target_net,
                                 gamma=GAMMA, tau=TAU, target_update_frequency=TARGET_UPDATE_FREQ, lr=LEARNING_RATE,
                                 log_folder=LOG_FOLDER, log_dropout_p=DQN_LOG_DROPOUT_P)

        self._env_number += 1

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
                dashboard.upload_exp_data(fields={agent_name: loss}, ep=episode, tick=None, measurement='loss')

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


if __name__ == '__main__':
    # Learner initialization
    learner = Learner()
    learner.launch()
