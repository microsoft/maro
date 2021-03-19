# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict

import torch
import torch.nn as nn

from .abs_block import AbsBlock


class FullyConnectedBlock(AbsBlock):
    """Fully connected network with optional batch normalization, activation and dropout components.

    Args:
        name (str): Network name.
        input_dim (int): Network input dimension.
        output_dim (int): Network output dimension.
        hidden_dims ([int]): Dimensions of hidden layers. Its length is the number of hidden layers.
        activation: A ``torch.nn`` activation type. If None, there will be no activation. Defaults to LeakyReLU.
        head (bool): If true, this block will be the top block of the full model and the top layer of this block
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
        input_dim: int,
        output_dim: int,
        hidden_dims: [int],
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
        self._input_dim = input_dim
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

        # build the net
        dims = [self._input_dim] + self._hidden_dims
        layers = [self._build_layer(in_dim, out_dim) for in_dim, out_dim in zip(dims, dims[1:])]
        # top layer
        layers.append(self._build_layer(dims[-1], self._output_dim, head=self._head))

        self._net = nn.Sequential(*layers)

        self._gradient_threshold = gradient_threshold
        if gradient_threshold is not None:
            for param in self._net.parameters():
                param.register_hook(lambda grad: torch.clamp(grad, -gradient_threshold, gradient_threshold))

        self._name = name

    def forward(self, x):
        out = self._net(x)
        if self._skip_connection:
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
