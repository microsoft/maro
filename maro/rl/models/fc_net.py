# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict

import torch
import torch.nn as nn


class FullyConnectedNet(nn.Module):
    """NN model to compute state or action values.

    Fully connected network with optional batch normalization, activation and dropout components.

    Args:
        name (str): Network name.
        input_dim (int): Network input dimension.
        output_dim (int): Network output dimension.
        hidden_dims ([int]): Dimensions of hidden layers. Its length is the number of hidden layers.
        activation: A ``torch.nn`` activation type. If None, there will be no activation. Defaults to LeakyReLU.
        softmax_enabled (bool): If true, the output of the net will be a softmax transformation of the top layer's
            output. Defaults to False.
        batch_norm_enabled (bool): If true, batch normalization will be performed at each layer.
        skip_connection_enabled (bool): If true, a skip connection will be built between the bottom (input) layer and
            top (output) layer. Defaults to False.
        dropout_p (float): Dropout probability. Defaults to None, in which case there is no drop-out.
        clip_value (float): Gradient clipping threshold. Defaults to None, in which case not gradient clipping is
            performed.
    """
    def __init__(
        self,
        name: str,
        input_dim: int,
        output_dim: int,
        hidden_dims: [int],
        activation=nn.LeakyReLU,
        softmax_enabled: bool = False,
        batch_norm_enabled: bool = False,
        skip_connection_enabled: bool = False,
        dropout_p: float = None,
        clip_value: float = None
    ):
        super().__init__()
        self._name = name
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims if hidden_dims is not None else []
        self._output_dim = output_dim

        # network features
        self._activation = activation
        self._softmax = nn.Softmax(dim=1) if softmax_enabled else None
        self._batch_norm_enabled = batch_norm_enabled
        self._dropout_p = dropout_p

        if skip_connection_enabled and input_dim != output_dim:
            raise ValueError(f"input and output dimensions must match if skip connection is enabled, "
                             f"got {input_dim} and {output_dim}")

        self._skip_connection_enabled = skip_connection_enabled

        # build the net
        self._layers = self._build_layers([input_dim] + self._hidden_dims)
        if len(self._hidden_dims) == 0:
            self._top_layer = nn.Linear(self._input_dim, self._output_dim)
        else:
            self._top_layer = nn.Linear(hidden_dims[-1], self._output_dim)
        self._net = nn.Sequential(*self._layers, self._top_layer)

        if clip_value is not None:
            for param in self._net.parameters():
                param.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    def forward(self, x):
        out = self._net(x)
        if self._skip_connection_enabled:
            out += x
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

    def _build_basic_layer(self, input_dim, output_dim):
        """Build basic layer.

        BN -> Linear -> Activation -> Dropout
        """
        components = []
        if self._batch_norm_enabled:
            components.append(("batch_norm", nn.BatchNorm1d(input_dim)))
        components.append(("linear", nn.Linear(input_dim, output_dim)))
        if self._activation is not None:
            components.append(("activation", self._activation()))
        if self._dropout_p:
            components.append(("dropout", nn.Dropout(p=self._dropout_p)))
        return nn.Sequential(OrderedDict(components))

    def _build_layers(self, layer_dims: []):
        """Build multi basic layer.

        BasicLayer1 -> BasicLayer2 -> ...
        """
        layers = []
        for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
            layers.append(self._build_basic_layer(input_dim, output_dim))
        return layers
