from datetime import datetime
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torchsummary import summary

from maro.utils import Logger, LogFormat

# Agent basic model of DDPG
class Actor(nn.Module):
    '''
    Actor network.
        Choose multi-layer full connection with dropout as the basic network architecture.
    '''
    def __init__(self, name: str, input_dim: int, hidden_dims: [int], output_dim: int, dropout_actor: float, log_enable: bool = True, log_folder: str = './'):
        '''
        Init Actor network.

        Args:
            name (str): Network name.
            input_dim (int): Network input dimension.
            hidden_dims ([int]): Network hiddenlayer dimension. The length of `hidden_dims` means the
                                hidden layer number, which requires larger than 1.
            output_dim (int): Network output dimension.
            dropout_p (float): Dropout parameter.
        '''
        super(Actor, self).__init__()
        assert(len(hidden_dims) > 1)
        self._name = name
        self._dropout_actor = dropout_actor
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._output_dim = output_dim
        self._layers = self._build_layers([input_dim] + hidden_dims)
        self._head = nn.Linear(hidden_dims[-1], output_dim)
        self._head.weight.data.uniform_(-3e-3, 3e-3)
        self._head.bias.data.uniform_(-3e-3, 3e-3)
        self._net = nn.Sequential(*self._layers, self._head)
        self._log_enable = log_enable
        # if self._log_enable:
            # self._model_summary_logger = Logger(tag=f'{self._name}.model_summary', format_=LogFormat.none, 
            #         dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
            # self._log_model_parameter_number()
            # self._model_summary_logger.debug(self._net)
            # self._model_parameters_logger = Logger(tag=f'{self._name}.model_parameters', format_=LogFormat.none, 
            #         dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
            # self.log_model_parameters(-1, -1)

    def _build_layers(self, layer_dims: []):
        '''
        Build multi basic layer.
            BasicLayer1 -> BasicLayer2 -> ...
        '''
        layers = []
        for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
            layers.append(nn.Sequential(nn.BatchNorm1d(input_dim),
                             nn.Linear(input_dim, output_dim),
                             nn.LeakyReLU(),
                             nn.Dropout(p=self._dropout_actor)))
        return layers

    def forward(self, x):
        values = self._net(x)
        return values
    
    @property
    def output_dim(self):
        return self._output_dim

    @property
    def input_dim(self):
        return self._input_dim

    def _log_model_parameter_number(self):
        total_parameter_number = sum([parameter.nelement() for parameter in self._net.parameters()])
        self._model_summary_logger.debug(f'total parameter number: {total_parameter_number}')

    def log_model_parameters(self, current_ep, learning_index):
        if self._log_enable:
            self._model_parameters_logger.debug(f'====================current_ep: {current_ep}, learning_index: {learning_index}=================')
            for name, param in self._net.named_parameters():
                self._model_parameters_logger.debug(name, param)

class Critic(nn.Module):
    '''
    Critic network.
        Choose multi-layer full connection with dropout as the basic network architecture.
    '''
    def __init__(self, name: str, input_dim: int, state_input_hidden_dims: [int], action_input_hidden_dims: [int], action_dim: int, dropout_critic: float, log_enable: bool = True, log_folder: str = './'):
        super(Critic, self).__init__()
        self._name = name
        self._dropout_critic = dropout_critic
        self._state_input_layers, self._action_input_layers = self._build_layers([input_dim] + state_input_hidden_dims, action_input_hidden_dims, action_dim)
        self._state_input_net = nn.Sequential(*self._state_input_layers)
        self._action_input_net = nn.Sequential(*self._action_input_layers)

        self._log_enable = log_enable
        # if self._log_enable:
            # self._model_summary_logger = Logger(tag=f'{self._name}.model_summary', format_=LogFormat.none, 
            #         dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
            # self._log_model_parameter_number()
            # self._model_summary_logger.debug(self._net)
            # self._model_parameters_logger = Logger(tag=f'{self._name}.model_parameters', format_=LogFormat.none, 
            #         dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
            # self.log_model_parameters(-1, -1)

    def forward(self, x, actions):
        x = self._state_input_net(x)
        values = self._action_input_net(torch.cat((x, actions.float()), 1))
        return values
    
    def _build_layers(self, state_input_hidden_dims: [], action_input_hidden_dims: [], action_dim: int):
        state_input_layers = []
        action_input_layers = []

        for input_dim, output_dim in zip(state_input_hidden_dims, state_input_hidden_dims[1:]):
            state_input_layers.append(nn.Sequential(nn.Linear(input_dim, output_dim), nn.LeakyReLU()))
        
        action_input_hidden_dims = [state_input_hidden_dims[-1] + action_dim] + action_input_hidden_dims[:-1]
        for input_dim, output_dim in zip(action_input_hidden_dims, action_input_hidden_dims[1:]):
            action_input_layers.append(nn.Sequential(nn.Linear(input_dim, output_dim),
                                                     nn.LeakyReLU(),
                                                     ))

        head = nn.Linear(action_input_hidden_dims[-1], 1)
        head.weight.data.uniform_(-3e-3, 3e-3)
        head.bias.data.uniform_(-3e-3, 3e-3)
        action_input_layers.append(head)

        return state_input_layers, action_input_layers

    def _log_model_parameter_number(self):
        total_parameter_number = sum([parameter.nelement() for parameter in self._net.parameters()])
        self._model_summary_logger.debug(f'total parameter number: {total_parameter_number}')

    def log_model_parameters(self, current_ep, learning_index):
        if self._log_enable:
            self._model_parameters_logger.debug(f'====================current_ep: {current_ep}, learning_index: {learning_index}=================')
            for name, param in self._net.named_parameters():
                self._model_parameters_logger.debug(name, param)

class DDPG(object):
    def __init__(self,
                 actor_policy_net: nn.Module, 
                 actor_target_net: nn.Module,
                 critic_policy_net: nn.Module, 
                 critic_target_net: nn.Module,
                 gamma: float,
                 tau: float,
                 critic_lr: float,
                 actor_lr: float,
                 theta: float,
                 sigma: float,
                 target_update_frequency: int,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 log_enable: bool = True, 
                 log_folder: str = './'):
        '''
        Args:
            actor_policy_net (nn.Module): actor net, which is used for choosing action.
            actor_target_net (nn.Module): actor target net, which is used for stablize trainning.
            critic_policy_net (nn.Module): actor net, which is used for evaluate actor.
            critic_target_net (nn.Module): actor target net, which is used for stablize trainning.
            gamma (float): Reward discount factor.
                         `expected_Q = reward + gamma * max(target_Q(next_state))`
            tau (float): Soft update parameter.
                         `target_θ = τ * policy_θ + (1 - τ) * target_θ`
            lr (float): Learning rate.
            device: Torch current device.
        '''
        super(DDPG, self).__init__()
        self._actor_policy_net = actor_policy_net.to(device)
        self._actor_policy_net.eval()
        self._actor_target_net = actor_target_net.to(device)
        self._actor_target_net.eval()

        self._critic_policy_net = critic_policy_net.to(device)
        self._critic_policy_net.eval()
        self._critic_target_net = critic_target_net.to(device)
        self._critic_target_net.eval()

        self._gamma = gamma
        self._tau = tau
        self._critic_lr = critic_lr
        self._actor_lr = actor_lr

        self._theta = theta
        self._sigma = sigma
        self._prev_noise = 0

        self._device = device
        self._actor_optimizer = optim.RMSprop(
            self._actor_policy_net.parameters(), lr=self._actor_lr)
        self._critic_optimizer = optim.RMSprop(
            self._critic_policy_net.parameters(), lr=self._critic_lr)
        self._learning_counter = 0
        self._target_update_frequency = target_update_frequency
        self._log_enable = log_enable

        if self._log_enable:
            self._logger = Logger(tag='DDPG', format_=LogFormat.simple, 
                dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
            self._actor_loss_logger = Logger(tag=f'actor_loss.{self._actor_policy_net._name}', format_=LogFormat.none, 
                    dump_folder=log_folder, dump_mode='w', extension_name='csv', auto_timestamp=False)
            self._actor_loss_logger.debug('episode,learning_index,loss')

            self._critic_loss_logger = Logger(tag=f'critic_loss.{self._critic_policy_net._name}', format_=LogFormat.none, 
                    dump_folder=log_folder, dump_mode='w', extension_name='csv', auto_timestamp=False)
            self._critic_loss_logger.debug('episode,learning_index,loss')

            self._Q_value_logger = Logger(tag=f'q_value.{self._critic_policy_net._name}', format_=LogFormat.none, 
                    dump_folder=log_folder, dump_mode='w', extension_name='csv', auto_timestamp=False)
            self._Q_value_logger.debug('episode, action, q_value')

    def choose_action(self, state: torch.Tensor, is_test: bool, current_ep:int) -> (bool, int):
        '''
        Args:
            state (tensor): Environment state, which is a tensor.
            eps (float): Epsilon, which is used for exploration.
            current_ep (int): Current episode, which is used for logging.

        Returns:
            (bool, int): is_random, action_index
        '''
        state = state.to(self._device)
        with torch.no_grad():
            action_value = self._actor_policy_net.forward(state).item()
        if not is_test:
            action_value = action_value + self._cal_ou_noise()
        return action_value

    def _cal_ou_noise(self):
        noise = self._prev_noise + self._theta * (-self._prev_noise) + self._sigma * np.random.normal(0, 0.3)
        self._prev_noise = noise
        return noise

    def learn(self, state_batch: torch.Tensor, action_batch: torch.Tensor, reward_batch: torch.Tensor, next_state_batch: torch.Tensor, current_ep: int) -> float:
        state_batch = state_batch.to(self._device)
        action_batch = action_batch.to(self._device)
        reward_batch = reward_batch.to(self._device)
        next_state_batch = next_state_batch.to(self._device)

        """ Critic """
        current_Q_values = self._critic_policy_net(state_batch, action_batch)
        target_actions = self._actor_target_net(next_state_batch)#.clamp(-self.action_lim, self.action_lim)
        next_Q_values = self._critic_target_net(next_state_batch, target_actions).detach()
        target_Q_values = reward_batch + (self._gamma * next_Q_values)

        critic_loss = F.mse_loss(current_Q_values, target_Q_values).mean()

        for action_index, q_values in zip(action_batch, current_Q_values):
            self._Q_value_logger.debug(f'{current_ep}, {action_index.item()}, {q_values.item()}')
        
        # Optimize the critic
        self._critic_policy_net.train()
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        # for param in self._Critic_policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self._critic_optimizer.step()
        self._critic_policy_net.eval()

        """ Actor """
        actor_loss = -self._critic_policy_net(state_batch, self._actor_policy_net(state_batch)).mean()

        # Optimize the actor
        self._actor_policy_net.train()
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        # for param in self._Actor_policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self._actor_optimizer.step()
        self._actor_policy_net.eval()

        '''update target critic and actor'''

        self._learning_counter += 1
        self._actor_policy_net.eval()
        self._critic_policy_net.eval()
        self._update_target_model()

        if self._log_enable:
            # self._critic_policy_net.log_model_parameters(current_ep, self._learning_counter)
            # self._actor_policy_net.log_model_parameters(current_ep, self._learning_counter)
            self._actor_loss_logger.debug(f'{current_ep},{self._learning_counter},actor,{actor_loss.item()}')
            self._critic_loss_logger.debug(f'{current_ep},{self._learning_counter},critic,{critic_loss.item()}')

        return critic_loss.item()

    @property
    def learning_index(self):
        return self._learning_counter

    def _update_target_model(self):
        '''
        Soft update target model.
            target_θ = τ*policy_θ + (1 - τ)*target_θ
        '''
        if self._learning_counter % self._target_update_frequency == 0:
            if self._log_enable:
                self._logger.debug(f'update model with {self._tau} * policy_param + {1 - self._tau} * target_param.')
            for policy_param, target_param in zip(self._actor_policy_net.parameters(), self._actor_target_net.parameters()):
                target_param.data = self._tau * policy_param.data + \
                    (1 - self._tau) * target_param.data
            for policy_param, target_param in zip(self._critic_policy_net.parameters(), self._critic_target_net.parameters()):
                target_param.data = self._tau * policy_param.data + \
                    (1 - self._tau) * target_param.data

    @property
    def policy_net(self):
        return self._actor_policy_net

    @property
    def target_net(self):
        return self._actor_target_net
    
    def dump_policy_net_parameters(self, dump_path):
        torch.save(self._actor_policy_net.state_dict(), dump_path)
