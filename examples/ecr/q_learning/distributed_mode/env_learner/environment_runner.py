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
from examples.ecr.q_learning.distributed_mode.env_learner.message_type import MsgType, MsgKey

from maro.simulator import Env
from maro.utils import SimpleExperiencePool, Logger, LogFormat, convert_dottable

from examples.ecr.q_learning.common.agent import Agent
from examples.ecr.q_learning.common.dqn import QNet, DQN
from examples.ecr.q_learning.common.state_shaping import StateShaping
from examples.ecr.q_learning.common.action_shaping import DiscreteActionShaping
from examples.ecr.q_learning.single_host_mode.runner import Runner
from examples.utils import log, get_proxy, generate_random_rgb

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
TRAIN_SEED = config.train.seed
TEST_SEED = config.test.seed
QNET_SEED = config.qnet.seed
RUNNER_LOG_ENABLE = config.log.runner.enable
AGENT_LOG_ENABLE = config.log.agent.enable
DQN_LOG_ENABLE = config.log.dqn.enable
DQN_LOG_DROPOUT_P = config.log.dqn.dropout_p
QNET_LOG_ENABLE = config.log.qnet.enable

COMPONENT = 'environment_runner'


class EnvRunner(Runner):
    def __init__(self, scenario: str, topology: str, max_tick: int, max_train_ep: int, max_test_ep: int,
                 eps_list: [float]):
        super().__init__(scenario, topology, max_tick, max_train_ep, max_test_ep, eps_list)
        self._agent_idx_list = self._env.agent_idx_list
        self._agent_2_learner = {self._agent_idx_list[i]: 'learner_' + str(i) for i in range(len(self._agent_idx_list))}
        self._proxy = get_proxy(COMPONENT, config, logger=self._logger)

    def launch(self, group_name, component_name):
        """
        setup the communication and trigger the training process.

        Args:
            group_name (str): identifier for the group of all distributed components
            component_name (str): unique identifier in the current group
        """
        self._proxy.join(group_name, component_name)
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
            agent.fulfill_cache(
                self._env.agent_idx_list, self._env.snapshot_list, current_ep=episode)
            self.send_experience(id_, episode)
            agent.clear_cache()

        self._env.reset()

    def send_net_parameters_to_learner(self):
        """
        Send initial net parameters to learners.
        """
        for agent_id in self._agent_idx_list:
            policy_net_params, target_net_params = self._get_net_parameters(agent_id)

            self._proxy.send(peer_name=self._agent_2_learner[agent_id], msg_type=MsgType.INITIAL_PARAMETERS,
                             msg_body={MsgKey.POLICY_NET_PARAMETERS: policy_net_params,
                                       MsgKey.TARGET_NET_PARAMETERS: target_net_params})

    def send_experience(self, agent_id, episode):
        """
        Send experiences from current episode to learner
        """
        agent_name = self._env.node_name_mapping['static'][agent_id]
        exp = self._agent_dict[agent_id].get_experience()

        self._proxy.send(peer_name=self._agent_2_learner[agent_id], msg_type=MsgType.STORE_EXPERIENCE,
                         msg_body={MsgKey.AGENT_ID: agent_id, MsgKey.EXPERIENCE: exp, MsgKey.EPISODE: episode,
                                   MsgKey.AGENT_NAME: agent_name})

    def send_env_checkout(self):
        """
        Send checkout message to learner
        """
        for agent_id in self._agent_idx_list:
            self._proxy.send(peer_name=self._agent_2_learner[agent_id], msg_type=MsgType.ENV_CHECKOUT,
                             msg_body={})

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
        if msg.body[MsgKey.POLICY_NET_PARAMETERS] != None:
            self._agent_dict[msg.body[MsgKey.AGENT_ID]].load_policy_net_parameters(
                msg.body[MsgKey.POLICY_NET_PARAMETERS])

    def force_sync(self):
        """
        Waiting for all agents have the updated policy net parameters, and message may 
        contain the policy net parameters.
        """
        pending_updated_agents = len(self._agent_idx_list)
        for msg in self._proxy.receive():
            if msg.type == MsgType.UPDATED_PARAMETERS:
                self.on_updated_parameters(msg)
                pending_updated_agents -= 1
            else:
                raise Exception(f'Unrecognized message type: {msg.type}')

            if not pending_updated_agents:
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
    component_name = '_'.join([COMPONENT, '0']) if 'INDEX' not in os.environ else '_'.join(
        [COMPONENT, os.environ['INDEX']])
    env_runner = EnvRunner(scenario=SCENARIO, topology=TOPOLOGY, max_tick=MAX_TICK,
                           max_train_ep=MAX_TRAIN_EP, max_test_ep=MAX_TEST_EP, eps_list=eps_list)
    env_runner.launch(os.environ['GROUP'], component_name)
