# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn


class MLPRepresentation(nn.Module):
    """
    Deep Q network.
        Choose multi-layer full connection with dropout as the basic network architecture.
    """

    def __init__(self, name: str, input_dim: int, hidden_dims: [int], output_dim: int, dropout_p: float):
        """
        Init deep Q network.

        Args:
            name (str): Network name.
            input_dim (int): Network input dimension.
            hidden_dims ([int]): Network hiddenlayer dimension. The length of `hidden_dims` means the
                                hidden layer number, which requires larger than 1.
            output_dim (int): Network output dimension.
            dropout_p (float): Dropout parameter.
        """
        super().__init__()
        self._name = name
        self._dropout_p = dropout_p
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims if hidden_dims is not None else []
        self._output_dim = output_dim
        self._layers = self._build_layers([input_dim] + self._hidden_dims)
        if len(self._hidden_dims) == 0:
            self._head = nn.Linear(self._input_dim, self._output_dim)
        else:
            self._head = nn.Linear(hidden_dims[-1], self._output_dim)
        self._net = nn.Sequential(*self._layers, self._head)

    def forward(self, x):
        return self._net(x).double()

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
        """
        Build basic layer.
            BN -> Linear -> LeakyReLU -> Dropout
        """
        return nn.Sequential(nn.Linear(input_dim, output_dim),
                             nn.LeakyReLU(),
                             nn.Dropout(p=self._dropout_p))

    def _build_layers(self, layer_dims: []):
        """
        Build multi basic layer.
            BasicLayer1 -> BasicLayer2 -> ...
        """
        layers = []
        for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
            layers.append(self._build_basic_layer(input_dim, output_dim))
        return layers
