# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Type

import torch
import torch.nn as nn


class FullyConnected(nn.Module):
    """Fully connected network with optional batch normalization, activation and dropout components.

    Args:
        input_dim (int): Network input dimension.
        output_dim (int): Network output dimension.
        hidden_dims (List[int]): Dimensions of hidden layers. Its length is the number of hidden layers. For example,
            `hidden_dims=[128, 256]` refers to two hidden layers with output dim of 128 and 256, respectively.
        activation (Optional[Type[torch.nn.Module], default=nn.ReLU): Activation class provided by ``torch.nn`` or a
            customized activation class. If None, there will be no activation.
        head (bool, default=False): If true, this block will be the top block of the full model and the top layer
            of this block will be the final output layer.
        softmax (bool, default=False): If true, the output of the net will be a softmax transformation of the top
            layer's output.
        batch_norm (bool, default=False): If true, batch normalization will be performed at each layer.
        skip_connection (bool, default=False): If true, a skip connection will be built between the bottom (input)
            layer and top (output) layer. Defaults to False.
        dropout_p (float, default=None): Dropout probability. If it is None, there will be no drop-out.
        gradient_threshold (float, default=None): Gradient clipping threshold. If it is None, no gradient clipping
            is performed.
        name (str, default=None): Network name.
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: Optional[Type[torch.nn.Module]] = nn.ReLU,
        head: bool = False,
        softmax: bool = False,
        batch_norm: bool = False,
        skip_connection: bool = False,
        dropout_p: float = None,
        gradient_threshold: float = None,
        name: str = "NONAME",
    ) -> None:
        super(FullyConnected, self).__init__()
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims if hidden_dims is not None else []
        self._output_dim = output_dim

        # network features
        self._activation = activation() if activation else None
        self._head = head
        self._softmax = nn.Softmax(dim=1) if softmax else None
        self._batch_norm = batch_norm
        self._dropout_p = dropout_p

        if skip_connection and input_dim != output_dim:
            raise ValueError(
                f"input and output dimensions must match if skip connection is enabled, "
                f"got {input_dim} and {output_dim}",
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._net(x.float())
        if self._skip_connection:
            out += x
        return self._softmax(out) if self._softmax else out

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _build_layer(self, input_dim: int, output_dim: int, head: bool = False) -> nn.Module:
        """Build a basic layer.

        BN -> Linear -> Activation -> Dropout
        """
        components: List[Tuple[str, nn.Module]] = []
        if self._batch_norm:
            components.append(("batch_norm", nn.BatchNorm1d(input_dim)))
        components.append(("linear", nn.Linear(input_dim, output_dim)))
        if not head and self._activation is not None:
            components.append(("activation", self._activation))
        if not head and self._dropout_p:
            components.append(("dropout", nn.Dropout(p=self._dropout_p)))
        return nn.Sequential(OrderedDict(components))
