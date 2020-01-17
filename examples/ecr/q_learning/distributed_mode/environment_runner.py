# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


# third party lib
import io
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import yaml

# private lib
from examples.ecr.q_learning.distributed_mode.message_type import MsgType, PayloadKey

from maro.simulator import Env
from maro.distributed import Proxy, Message
from maro.utils import SimpleExperiencePool, Logger, LogFormat, convert_dottable

from examples.ecr.q_learning.common.agent import Agent
from examples.ecr.q_learning.common.dqn import QNet, DQN
from examples.ecr.q_learning.common.state_shaping import StateShaping
from examples.ecr.q_learning.common.action_shaping import DiscreteActionShaping
from examples.ecr.q_learning.single_host_mode.runner import Runner
from examples.utils import log, get_peers

CONFIG_PATH = os.environ.get('CONFIG') or 'config.yml'

with io.open(CONFIG_PATH, 'r') as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

LOG_FOLDER = os.path.join(os.getcwd(), 'log', f"{datetime.now().strftime('%Y%m%d')}", config.experiment_name)
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

with io.open(os.path.join(LOG_FOLDER, 'config.yml'), 'w', encoding='utf8') as out_file:
    yaml.safe_dump(raw_config, out_file)

COMPONENT_TYPE = os.environ['COMPTYPE']
COMPONENT_ID = os.environ.get('COMPID', None)
COMPONENT_NAME = '.'.join([COMPONENT_TYPE,COMPONENT_ID])
SCENARIO = config.env.scenario
TOPOLOGY = config.env.topology
MAX_TICK = config.env.max_tick
MAX_TRAIN_EP = config.train.max_ep
MAX_TEST_EP = config.test.max_ep
MAX_EPS = config.train.exploration.max_eps
PHASE_SPLIT_POINT = config.train.exploration.phase_split_point
FIRST_PHASE_REDUCE_PROPORTION = config.train.exploration.first_phase_reduce_proportion
TARGET_UPDATE_FREQ = config.train.dqn.target_update_frequency
LEARNING_RATE = config.train.dqn.lr
DROPOUT_P = config.train.dqn.dropout_p
GAMMA = config.train.dqn.gamma  # Reward decay
TAU = config.train.dqn.tau  # Soft update
BATCH_NUM = config.train.batch_num
BATCH_SIZE = config.train.batch_size
MIN_TRAIN_EXP_NUM = config.train.min_train_experience_num  # when experience num is less than this num, agent will not train model
REWARD_SHAPING = config.train.reward_shaping
TRAIN_SEED = config.train.seed + int(COMPONENT_ID if COMPONENT_ID is not None else 0)
TEST_SEED = config.test.seed
QNET_SEED = config.qnet.seed
RUNNER_LOG_ENABLE = config.log.runner.enable
AGENT_LOG_ENABLE = config.log.agent.enable
DQN_LOG_ENABLE = config.log.dqn.enable
DQN_LOG_DROPOUT_P = config.log.dqn.dropout_p
QNET_LOG_ENABLE = config.log.qnet.enable


class EnvRunner(Runner):
    def __init__(self, scenario: str, topology: str, max_tick: int, max_train_ep: int, max_test_ep: int,
                 eps_list: [float], log_enable: bool = True, dashboard_enable: bool = True):
        super().__init__(scenario, topology, max_tick, max_train_ep, max_test_ep, eps_list,
                         log_enable=log_enable, dashboard_enable=dashboard_enable)
        self._agent_idx_list = self._env.agent_idx_list
        self._agent2learner = {self._agent_idx_list[i]: 'learner.' + str(i) for i in range(len(self._agent_idx_list))}
        self._proxy = Proxy(group_name=os.environ['GROUP'],
                            component_name=COMPONENT_NAME,
                            peer_name_list=get_peers(COMPONENT_TYPE, config.distributed),
                            redis_address=(config.redis.host, config.redis.port),
                            logger=self._logger)

        if log_enable:
            self._logger = Logger(tag=COMPONENT_NAME, format_=LogFormat.simple,
                                  dump_folder=LOG_FOLDER, dump_mode='w', auto_timestamp=False)
            self._performance_logger = Logger(tag=f'{COMPONENT_NAME}.performance', format_=LogFormat.none,
                                              dump_folder=LOG_FOLDER, dump_mode='w', extension_name='csv',
                                              auto_timestamp=False)

    def launch(self):
        """
        setup the communication and trigger the training process.
        """
        self._proxy.join()
        self.send_net_parameters_to_learner()
        pbar = tqdm(range(MAX_TRAIN_EP))
        for ep in pbar:
            pbar.set_description('train episode')
            self.start(ep)
            self.force_sync()

        self.send_env_checkout()
        self._test()

    def start(self, episode):
        """
        Interaction with the environment, and send experiences get from the current episode to learner.
        Args:
            episode: int
        """
        self._set_seed(TRAIN_SEED + episode)

        _, decision_event, is_done = self._env.step(None)

        while not is_done:
            action = self._agent_dict[decision_event.port_idx].choose_action(
                decision_event=decision_event, eps=self._eps_list[episode], current_ep=episode)
            _, decision_event, is_done = self._env.step(action)

        self._print_summary(ep=episode, is_train=True)

        for id_, agent in self._agent_dict.items():
            agent.calculate_offline_rewards(episode)
            self.send_experience(id_, episode)

        self._env.reset()

    def send_net_parameters_to_learner(self):
        """
        Send initial net parameters to learners.
        """
        for agent_id in self._agent_idx_list:
            policy_net_params, target_net_params = self._get_net_parameters(agent_id)
            message = Message(type=MsgType.INITIAL_PARAMETERS, source=self._proxy.name,
                              destination=self._agent2learner[agent_id],
                              payload={PayloadKey.POLICY_NET_PARAMETERS: policy_net_params,
                                    PayloadKey.TARGET_NET_PARAMETERS: target_net_params})
            self._proxy.send(message)

    def send_experience(self, agent_id, episode):
        """
        Send experiences from current episode to learner
        """
        agent_name = self._env.node_name_mapping['static'][agent_id]
        exp = self._agent_dict[agent_id].get_experience()
        message = Message(type=MsgType.STORE_EXPERIENCE, source=self._proxy.name,
                          destination=self._agent2learner[agent_id],
                          payload={PayloadKey.AGENT_ID: agent_id, PayloadKey.EXPERIENCE: exp, PayloadKey.EPISODE: episode,
                                PayloadKey.AGENT_NAME: agent_name})
        self._proxy.send(message)

    def send_env_checkout(self):
        """
        Send checkout message to learner
        """
        for agent_id in self._agent_idx_list:
            message = Message(type=MsgType.ENV_CHECKOUT, source=self._proxy.name,
                              destination=self._agent2learner[agent_id])
            self._proxy.send(message)

    def _get_net_parameters(self, agent_id):
        """
        Get the policy net parameters and target net parameters
        Args:
            agent_id: str

        Return
            params: list of tuples(name, input_dim, hidden_dims, output_dim, dropout_p)
        """
        params = []
        for which in {'policy', 'target'}:
            net = getattr(self._agent_dict[agent_id].algorithm, f'{which}_net')
            params.append(
                (f'{self._port_idx2name[agent_id]}.{which}', net.input_dim, [256, 128, 64], net.output_dim, DROPOUT_P))

        return params

    def on_updated_parameters(self, msg):
        """
        Handles policy net parameters from learner. This message should contain the agent id and policy net parameters.

        Load policy net parameters for the given agent's algorithm
        """
        if msg.payload[PayloadKey.POLICY_NET_PARAMETERS] is not None:
            self._agent_dict[msg.payload[PayloadKey.AGENT_ID]].load_policy_net_parameters(
                msg.payload[PayloadKey.POLICY_NET_PARAMETERS])

    def force_sync(self):
        """
        Waiting for all agents have the updated policy net parameters, and message may
        contain the policy net parameters.
        """
        pending_learner_count = len(self._agent_idx_list)
        for msg in self._proxy.receive():
            if msg.type == MsgType.UPDATED_PARAMETERS:
                self.on_updated_parameters(msg)
                pending_learner_count -= 1
            elif msg.type == MsgType.NO_UPDATED_PARAMETERS:
                pending_learner_count -= 1
            else:
                raise Exception(f'Unrecognized message type: {msg.type}')

            if pending_learner_count == 0:
                break


if __name__ == '__main__':
    # Calculate the epsilon value
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

    # EnvRunner initialization
    env_runner = EnvRunner(scenario=SCENARIO, topology=TOPOLOGY, max_tick=MAX_TICK,
                           max_train_ep=MAX_TRAIN_EP, max_test_ep=MAX_TEST_EP, eps_list=eps_list)
    env_runner.launch()
