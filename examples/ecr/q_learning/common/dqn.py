# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from datetime import datetime
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary

from maro.utils import Logger, LogFormat


class QNet(nn.Module):
    '''
    Deep Q network.
        Choose multi-layer full connection with dropout as the basic network architecture.
    '''

    def __init__(self, name: str, input_dim: int, hidden_dims: [int], output_dim: int, dropout_p: float,
                 log_folder: str = None):
        '''
        Init deep Q network.

        Args:
            name (str): Network name.
            input_dim (int): Network input dimension.
            hidden_dims ([int]): Network hiddenlayer dimension. The length of `hidden_dims` means the
                                hidden layer number, which requires larger than 1.
            output_dim (int): Network output dimension.
            dropout_p (float): Dropout parameter.
        '''
        super(QNet, self).__init__()
        assert (len(hidden_dims) > 1)
        self._name = name
        self._dropout_p = dropout_p
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._output_dim = output_dim
        self._layers = self._build_layers([input_dim] + hidden_dims)
        self._head = nn.Linear(hidden_dims[-1], output_dim)
        self._net = nn.Sequential(*self._layers, self._head)
        self._log_enable = False if log_folder is None else True
        if self._log_enable:
            self._model_summary_logger = Logger(tag=f'{self._name}.model_summary', format_=LogFormat.none,
                                                dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
            self._log_model_parameter_number()
            self._model_summary_logger.debug(self._net)
            self._model_parameters_logger = Logger(tag=f'{self._name}.model_parameters', format_=LogFormat.none,
                                                   dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
            self.log_model_parameters(-1, -1)

    def forward(self, x):
        q_values_batch = self._net(x)
        return q_values_batch

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def name(self):
        return self._name

    @property
    def output_dim(self):
        return self._output_dim

    def _build_basic_layer(self, input_dim, output_dim):
        '''
        Build basic layer.
            BN -> Linear -> LeakyReLU -> Dropout
        '''
        return nn.Sequential(nn.BatchNorm1d(input_dim),
                             nn.Linear(input_dim, output_dim),
                             nn.LeakyReLU(),
                             nn.Dropout(p=self._dropout_p))

    def _build_layers(self, layer_dims: []):
        '''
        Build multi basic layer.
            BasicLayer1 -> BasicLayer2 -> ...
        '''
        layers = []
        for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
            layers.append(self._build_basic_layer(input_dim, output_dim))
        return layers

    def _log_model_parameter_number(self):
        total_parameter_number = sum([parameter.nelement() for parameter in self._net.parameters()])
        self._model_summary_logger.debug(f'total parameter number: {total_parameter_number}')

    def log_model_parameters(self, current_ep, learning_index):
        if self._log_enable:
            self._model_parameters_logger.debug(
                f'====================current_ep: {current_ep}, learning_index: {learning_index}=================')
            for name, param in self._net.named_parameters():
                self._model_parameters_logger.debug(name, param)


class DQN(object):
    def __init__(self,
                 policy_net: nn.Module,
                 target_net: nn.Module,
                 gamma: float,
                 tau: float,
                 lr: float,
                 target_update_frequency: int,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 log_folder: str = None, log_dropout_p: float = 0.0,
                 dashboard: object = None):
        '''
        Args:
            policy_net (nn.Module): Policy Q net, which is used for choosing action.
            target_net (nn.Module): Target Q net, which is used for evaluating next state.
            gamma (float): Reward discount factor.
                         `expected_Q = reward + gamma * max(target_Q(next_state))`
            tau (float): Soft update parameter.
                         `target_θ = τ * policy_θ + (1 - τ) * target_θ`
            lr (float): Learning rate.
            device: Torch current device.
        '''
        super(DQN, self).__init__()
        self._policy_net = policy_net.to(device)
        self._policy_net.eval()
        self._target_net = target_net.to(device)
        self._target_net.eval()
        self._gamma = gamma
        self._tau = tau
        self._lr = lr
        self._device = device
        self._optimizer = optim.RMSprop(
            self._policy_net.parameters(), lr=self._lr)
        self._learning_counter = 0
        self._target_update_frequency = target_update_frequency
        self._log_enable = False if log_folder is None else True
        self._log_dropout_p = log_dropout_p
        self._log_folder = log_folder
        self._dashboard = dashboard
        if self._log_enable:
            self._logger = Logger(tag='dqn', format_=LogFormat.simple,
                                  dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
            self._loss_logger = Logger(tag=f'{self._policy_net.name}.loss', format_=LogFormat.none,
                                       dump_folder=log_folder, dump_mode='w', extension_name='csv',
                                       auto_timestamp=False)
            self._loss_logger.debug('episode,learning_index,loss')
            self._q_curve_logger = Logger(tag=f'{self._policy_net.name}.q_curve', format_=LogFormat.none,
                                          dump_folder=log_folder, dump_mode='w', extension_name='csv',
                                          auto_timestamp=False)
            self._q_curve_logger.debug(
                'episode,learning_index,' + ','.join([str(i) for i in range(self._policy_net.output_dim)]))

    def choose_action(self, state: torch.Tensor, eps: float, current_ep: int, current_tick: int) -> (bool, int):
        '''
        Args:
            state (tensor): Environment state, which is a tensor.
            eps (float): Epsilon, which is used for exploration.
            current_ep (int): Current episode, which is used for logging.
            current_tick (int): Current tick, which is used for dashboard.
            is_train (bool): True is training, False is testing, which is used for dashboard.
            trained_ep (int): Trained ep, if is test, which is used for dashboard.

        Returns:
            (bool, int): is_random, action_index
        '''
        state = state.to(self._device)
        sample = random.random()
        if sample > eps:
            with torch.no_grad():
                q_values_batch = self._policy_net(state)
                if self._log_enable:
                    sample = random.random()
                    if sample > self._log_dropout_p:
                        for q_values in q_values_batch:
                            self._q_curve_logger.debug(f'{current_ep},{self._learning_counter},' + ','.join(
                                [str(q_value.item()) for q_value in q_values]))
                if self._dashboard is not None:
                    dashboard_ep = current_ep
                    if not self._dashboard.dynamic_info['is_train']:
                        dashboard_ep += self._dashboard.static_info['max_train_ep']
                    for q_values in q_values_batch:
                        for i in range(len(q_values)):
                            scalars = {self._policy_net.name: q_values[i].item(), 'action': i, 'tick': current_tick}
                            self._dashboard.upload_exp_data(scalars, dashboard_ep, None, 'q_value')
                action = q_values_batch.max(1)[1][0].item()
                return False, action
        else:
            return True, random.choice(range(self._policy_net.output_dim))

    def learn(self, state_batch: torch.Tensor, action_batch: torch.Tensor, reward_batch: torch.Tensor,
              next_state_batch: torch.Tensor, current_ep: int) -> float:
        state_batch = state_batch.to(self._device)
        action_batch = action_batch.to(self._device)
        reward_batch = reward_batch.to(self._device)
        next_state_batch = next_state_batch.to(self._device)

        self._policy_net.train()
        policy_state_action_values = self._policy_net(
            state_batch).gather(1, action_batch.long())
        # self._logger.debug(f'policy state action values: {policy_state_action_values}')

        target_next_state_values = self._target_net(
            next_state_batch).max(1)[0].view(-1, 1).detach()
        # self._logger.debug(f'target next state values: {target_next_state_values}')

        expected_state_action_values = reward_batch + \
                                       self._gamma * target_next_state_values

        # self._logger.debug(f'expected state action values: {expected_state_action_values}')

        loss = F.smooth_l1_loss(
            policy_state_action_values, expected_state_action_values).mean()

        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()
        self._learning_counter += 1
        self._policy_net.eval()
        self._soft_update_target_net_parameters()

        if self._log_enable:
            sample = random.random()
            if sample > self._log_dropout_p:
                self._policy_net.log_model_parameters(current_ep, self._learning_counter)
                self._target_net.log_model_parameters(current_ep, self._learning_counter)
                self._loss_logger.debug(f'{current_ep},{self._learning_counter},{loss.item()}')
        return loss.item()

    @property
    def learning_index(self):
        return self._learning_counter

    def _soft_update_target_net_parameters(self):
        '''
        Soft update target model.
            target_θ = τ*policy_θ + (1 - τ)*target_θ
        '''
        if self._learning_counter % self._target_update_frequency == 0:
            if self._log_enable:
                self._logger.debug(
                    f'{self._target_net.name} update model with {self._tau} * policy_param + {1 - self._tau} * target_param.')
            for policy_param, target_param in zip(self._policy_net.parameters(), self._target_net.parameters()):
                target_param.data = self._tau * policy_param.data + \
                                    (1 - self._tau) * target_param.data

    @property
    def policy_net(self):
        return self._policy_net

    @property
    def target_net(self):
        return self._target_net

    def load_policy_net_parameters(self, policy_param):
        self._policy_net.load_state_dict(policy_param)

    def dump_policy_net_parameters(self, dump_path):
        torch.save(self._policy_net.state_dict(), dump_path)


