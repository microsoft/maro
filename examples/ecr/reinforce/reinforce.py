from torch.distributions import Categorical

import random
import torch
import torch.nn as nn

from maro.utils import Logger, LogFormat

class ActorNet(nn.Module):

    def __init__(self, name: str, input_dim: int, hidden_dims: [int], output_dim: int, log_enable: bool = True, log_folder: str = './', init_w: float = 1e-3):
        super(ActorNet, self).__init__()
        assert(len(hidden_dims) > 1)
        self._name = name
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._output_dim = output_dim
        self._num_layers = len(self._hidden_dims) + 1
        self._log_enable = log_enable
        
        
        layer_sizes = [input_dim] + self._hidden_dims
        layers = []
        for i in range(self._num_layers - 1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                       nn.Tanh()
                       ]
        self._hidden_layer = nn.Sequential(*layers)

        self._last_layer = nn.Linear(layer_sizes[-1], output_dim)
        if init_w:
            self._last_layer.weight.data.uniform_(-init_w, init_w)
            self._last_layer.bias.data.uniform_(-init_w, init_w)

        # TODO: dim=1 for batch forward; dim=0 if only one
        self._soft_max = nn.Softmax()

    def forward(self, x):
        x = self._hidden_layer(x)
        x = self._last_layer(x)
        return self._soft_max(x)


    @property
    def name(self):
        return self._name

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    def _log_model_parameter_number(self):
        total_parameter_number = sum([parameter.nelement() for parameter in self._net.parameters()])
        self._model_summary_logger.debug(f'total parameter number: {total_parameter_number}')

    def log_model_parameters(self, current_ep, learning_index):
        if self._log_enable:
            self._model_parameters_logger.debug(f'====================current_ep: {current_ep}, learning_index: {learning_index}=================')
            for name, param in self._net.named_parameters():
                self._model_parameters_logger.debug(name, param)

class Reinforce(object):
    def __init__(self,
                 policy_net: nn.Module,
                 lr: float,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 log_enable: bool = True, log_folder: str = './',
                 log_dropout_p: float = 0.0,
                 dashboard_enable: bool = True, dashboard: object = None):
        self._policy_net = policy_net.to(device)
        self._policy_net.eval()
        self._lr = lr
        self._device = device
        self._learning_counter = 0
        self._log_enable = log_enable
        self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=self._lr, weight_decay=1e-5)
        if self._log_enable:
            self._logger = Logger(tag='reinforce',
                dump_folder=log_folder, dump_mode='w', auto_timestamp=False)
            self._loss_logger = Logger(tag=f'{self._policy_net.name}.loss',
                    dump_folder=log_folder, dump_mode='w', extension_name='csv', auto_timestamp=False)
            self._loss_logger.debug('episode,learning_index,loss')

    def choose_action(self, state: torch.Tensor, current_ep:int) -> (bool, int):
        '''
        Args:
            state (tensor): Environment state, which is a tensor.
            current_ep (int): Current episode, which is used for logging.

        Returns:
            (bool, int): is_random, action_index
        '''
        with torch.no_grad():
            probs = self._policy_net.forward(torch.FloatTensor(state).to(self._device))
            m = Categorical(probs)
            action = m.sample()
            # action = probs.argmax()
            return action.item()


    def learn(self, state_batch: torch.Tensor, action_batch: torch.Tensor, reward_batch: torch.Tensor, next_state_batch: torch.Tensor, current_ep: int) -> float:
        state_batch = state_batch.to(self._device)
        action_batch = action_batch.to(self._device)
        reward_batch = reward_batch.to(self._device)
        next_state_batch = next_state_batch.to(self._device)

        self._policy_net.train()
        # self._logger.debug(f'policy state action values: {policy_state_action_values}')

        # self._logger.debug(f'target next state values: {target_next_state_values}')

        # self._logger.debug(f'expected state action values: {expected_state_action_values}')
        loss = 0
        for state, action, reward in zip(state_batch, action_batch, reward_batch):
            probs = self._policy_net.forward(state)
            m = Categorical(probs)
            loss += -m.log_prob(action) * reward
        loss = loss.sum()

        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()
        self._learning_counter += 1
        self._policy_net.eval()

        if self._log_enable:
            sample = random.random()
            self._policy_net.log_model_parameters(current_ep, self._learning_counter)
            self._loss_logger.debug(f'{current_ep},{self._learning_counter},{loss.item()}')

        return loss.item()

    @property
    def learning_index(self):
        return self._learning_counter

    @property
    def policy_net(self):
        return self._policy_net

    def dump_policy_net_parameters(self, dump_path):
        torch.save(self._policy_net.state_dict(), dump_path)
