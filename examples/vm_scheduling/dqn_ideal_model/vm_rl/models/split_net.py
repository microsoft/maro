# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict

import torch
import torch.nn as nn

from maro.rl import AbsBlock


class SplitNet(AbsBlock):
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
        tick_input_dim: int,
        pm_input_dim: int,
        vm_input_dim: int,
        output_dim: int,
        hidden_dims: [int],
        activation=nn.LeakyReLU,
        is_head: bool = False,
        softmax: bool = False,
        batch_norm: bool = False,
        skip_connection: bool = False,
        dropout_p: float = None,
        gradient_threshold: float = None,
        name: str = None
    ):
        super().__init__()
        self._pm_num = pm_num
        self._tick_input_dim = tick_input_dim
        self._pm_input_dim = pm_input_dim
        self._vm_input_dim = vm_input_dim
        self._hidden_dims = hidden_dims if hidden_dims is not None else []
        self._output_dim = output_dim

        # network features
        self._activation = activation
        self._is_head = is_head
        self._softmax = nn.Softmax(dim=1) if softmax else None
        self._batch_norm = batch_norm
        self._dropout_p = dropout_p

        if skip_connection and input_dim != output_dim:
            raise ValueError(
                f"input and output dimensions must match if skip connection is enabled, "
                f"got {input_dim} and {output_dim}"
            )

        self._skip_connection = skip_connection

        # build the tick net
        tick_dims = [self._tick_input_dim] + self._hidden_dims[:-1]
        tick_layers = [self._build_layer(in_dim, out_dim) for in_dim, out_dim in zip(tick_dims, tick_dims[1:])]
        self._tick_net = nn.Sequential(*tick_layers).double()
        # build the pm net
        pm_dims = [self._pm_input_dim] + self._hidden_dims[:-1]
        pm_layers = [self._build_layer(in_dim, out_dim) for in_dim, out_dim in zip(pm_dims, pm_dims[1:])]
        self._pm_net = nn.Sequential(*pm_layers).double()
        # build the vm net
        vm_dims = [self._vm_input_dim] + self._hidden_dims[:-1]
        vm_layers = [self._build_layer(in_dim, out_dim) for in_dim, out_dim in zip(vm_dims, vm_dims[1:])]
        self._vm_net = nn.Sequential(*vm_layers).double()
        # top layer
        layers = []
        layers.append(self._build_layer(tick_dims[-1] + pm_dims[-1] * self._pm_num + vm_dims[-1], self._hidden_dims[-1]))
        layers.append(self._build_layer(self._hidden_dims[-1], self._output_dim, is_head=self._is_head))
        self._net = nn.Sequential(*layers).double()

        self._gradient_threshold = gradient_threshold
        if gradient_threshold is not None:
            for param in self._net.parameters():
                param.register_hook(lambda grad: torch.clamp(grad, -gradient_threshold, gradient_threshold))

        self._name = name
        self._initialize_weights()

    def forward(self, x):
        st = 1

        feature = []
        
        tick_feature = self._tick_net(x[:, 0].unsqueeze(1))
        feature.append(tick_feature)
        
        for _ in range(self._pm_num):
            pm_input = x[:, st:st+self._pm_input_dim]
            pm_feature = self._pm_net(pm_input)
            feature.append(pm_feature)
            st += self._pm_input_dim
        
        vm_input = x[:, st:st+self._vm_input_dim]
        vm_feature = self._vm_net(vm_input)
        feature.append(vm_feature)
        
        feature = torch.cat(feature, dim=1)

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

    def _build_layer(self, input_dim, output_dim, is_head: bool = False):
        """Build basic layer.

        BN -> Linear -> Activation -> Dropout
        """
        components = []
        if self._batch_norm:
            components.append(("batch_norm", nn.BatchNorm1d(input_dim)))
        components.append(("linear", nn.Linear(input_dim, output_dim)))
        if not is_head and self._activation is not None:
            components.append(("activation", self._activation()))
        if not is_head and self._dropout_p:
            components.append(("dropout", nn.Dropout(p=self._dropout_p)))
        return nn.Sequential(OrderedDict(components))

    def _initialize_weights(self):
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
