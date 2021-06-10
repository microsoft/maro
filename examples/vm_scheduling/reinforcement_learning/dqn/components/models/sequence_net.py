# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pdb
from typing import List
from collections import OrderedDict

import torch
import torch.nn as nn

from maro.rl import AbsBlock


class SequenceNet(AbsBlock):
    """Fully connected network with optional batch normalization, activation and dropout components.

    Args:
        name (str): Network name.
        input_dim (int): Network input dimension.
        output_dim (int): Network output dimension.
        hidden_dims ([int]): Dimensions of hidden layers. Its length is the number of hidden layers.
        activation: A ``torch.nn`` activation type. If None, there will be no activation. Defaults to LeakyReLU.
        is_head (bool): If true, this block will be the top block of the full model and the top layer of this block
            will be the final output layer. Defaults to False.
        softmax (bool): If true, the output of the net will be a softmax transformation of the top layer's
            output. Defaults to False.
        batch_norm (bool): If true, batch normalization will be performed at each layer.
        skip_connection (bool): If true, a skip connection will be built between the bottom (input) layer and
            top (output) layer. Defaults to False.
        dropout_p (float): Dropout probability. Defaults to None, in which case there is no drop-out.
        gradient_threshold (float): Gradient clipping threshold. Defaults to None, in which case not gradient clipping
            is performed.
    """
    def __init__(
        self,
        pm_num: int,
        pm_input_dim: int,
        pm_window_size: int,
        vm_input_dim: int,
        vm_window_size: int,
        output_dim: int,
        hidden_dims: List[int],
        activation=nn.LeakyReLU,
        head: bool = False,
        softmax: bool = False,
        batch_norm: bool = False,
        skip_connection: bool = False,
        dropout_p: float = None,
        gradient_threshold: float = None,
        name: str = None
    ):
        super().__init__()
        self._pm_num = pm_num
        self._pm_input_dim = pm_input_dim
        self._pm_window_size = pm_window_size
        self._pm_state_dim = self._pm_num * self._pm_input_dim * self._pm_window_size
        self._vm_input_dim = vm_input_dim
        self._vm_window_size = vm_window_size
        self._vm_state_dim = self._vm_input_dim * self._vm_window_size
        self._hidden_dims = hidden_dims if hidden_dims is not None else []
        self._output_dim = output_dim

        # network features
        self._activation = activation
        self._head = head
        self._softmax = nn.Softmax(dim=1) if softmax else None
        self._batch_norm = batch_norm
        self._dropout_p = dropout_p

        if skip_connection and input_dim != output_dim:
            raise ValueError(
                f"input and output dimensions must match if skip connection is enabled, "
                f"got {input_dim} and {output_dim}"
            )

        self._skip_connection = skip_connection

        # build the pm sequence net
        pm_dims = [self._pm_input_dim*self._pm_num] + self._hidden_dims[:2]
        pm_layers = [self._build_layer(in_dim, out_dim) for in_dim, out_dim in zip(pm_dims, pm_dims[1:])]
        self._pm_info_net = nn.Sequential(*pm_layers).double()
        self._pm_sequence_rnn = nn.LSTM(
            input_size=self._hidden_dims[0], hidden_size=self._hidden_dims[1],
            num_layers=1, bidirectional=False, batch_first=True
        ).double()
        # build the vm sequence net
        vm_dims = [self._vm_input_dim] + self._hidden_dims[:1]
        vm_layers = [self._build_layer(in_dim, out_dim) for in_dim, out_dim in zip(vm_dims, vm_dims[1:])]
        self._vm_info_net = nn.Sequential(*vm_layers).double()
        self._vm_sequence_rnn = nn.LSTM(
            input_size=self._hidden_dims[0], hidden_size=self._hidden_dims[1],
            num_layers=1, bidirectional=False, batch_first=True
        ).double()
        # top layer
        layers = []
        layers.append(self._build_layer(2 * self._hidden_dims[1], self._hidden_dims[-1]))
        layers.append(self._build_layer(self._hidden_dims[-1], self._output_dim, head=self._head))
        self._net = nn.Sequential(*layers).double()

        self._gradient_threshold = gradient_threshold
        if gradient_threshold is not None:
            for param in self._net.parameters():
                param.register_hook(lambda grad: torch.clamp(grad, -gradient_threshold, gradient_threshold))

        self._name = name

    def forward(self, x):
        pm_info_input = x[:, :self._pm_state_dim].view(-1, self._pm_window_size, self._pm_num * self._pm_input_dim)
        pm_info_feature = self._pm_info_net(pm_info_input)
        # self._pm_sequence_rnn.flatten_parameters()
        # pm_sequence_feature, _ = self._pm_sequence_rnn(pm_info_feature)

        vm_info_input = x[:, -self._vm_state_dim:].view(-1, self._vm_window_size, self._vm_input_dim)
        vm_info_feature = self._vm_info_net(vm_info_input)
        self._vm_sequence_rnn.flatten_parameters()
        vm_sequence_feature, _ = self._vm_sequence_rnn(vm_info_feature)
        # feature = torch.cat([pm_sequence_feature[:, -1, :], vm_sequence_feature[:, -1, :]], dim=1)
        feature = torch.cat([pm_info_feature.squeeze(1), vm_sequence_feature[:, -1, :]], dim=1)

        out = self._net(feature)

        return self._softmax(out) if self._softmax else out

    @property
    def name(self):
        return self._name

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    def _build_layer(self, input_dim, output_dim, head: bool = False):
        """Build basic layer.

        BN -> Linear -> Activation -> Dropout
        """
        components = []
        if self._batch_norm:
            components.append(("batch_norm", nn.BatchNorm1d(input_dim)))
        components.append(("linear", nn.Linear(input_dim, output_dim)))
        if not head and self._activation is not None:
            components.append(("activation", self._activation()))
        if not head and self._dropout_p:
            components.append(("dropout", nn.Dropout(p=self._dropout_p)))
        return nn.Sequential(OrderedDict(components))
