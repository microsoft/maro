# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn


class MLPPolicyNet(nn.Module):
    """NN model to compute action distributions given states.

    Args:
        name (str): Network name.
        input_dim (int): Network input dimension.
        hidden_dims ([int]): Network hidden layer dimension. The length of ``hidden_dims`` means the
                            hidden layer number, which requires larger than 1.
        output_dim (int): Network output dimension.
        init_w (float): If not None, [-init_w, init_w] will be the range from which the initial network parameters
            will be uniformly drawn.
    """
    def __init__(
        self, name: str, input_dim: int, hidden_dims: [int], output_dim: int, init_w: float = 1e-3
    ):
        super().__init__()
        assert len(hidden_dims) > 1
        self._name = name
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._output_dim = output_dim
        self._num_layers = len(self._hidden_dims) + 1

        layer_sizes = [input_dim] + self._hidden_dims
        layers = []
        for i in range(self._num_layers - 1):
            layers += [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                nn.Tanh()
            ]
        self._hidden_layer = nn.Sequential(*layers)

        self._last_layer = nn.Linear(layer_sizes[-1], output_dim)
        if init_w is not None:
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
