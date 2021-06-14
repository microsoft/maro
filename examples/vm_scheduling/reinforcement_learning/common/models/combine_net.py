# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch.nn as nn

from maro.rl import FullyConnectedBlock


class CombineNet(FullyConnectedBlock):
    """Fully connected network with optional batch normalization, activation and dropout components.

    Args:
        name (str): Network name.
        input_dim (int): Network input dimension.
        output_dim (int): Network output dimension.
        hidden_dims ([int]): Dimensions of hidden layers. Its length is the number of hidden layers.
        activation: A string indicatinfg an activation class provided by ``torch.nn`` or a custom activation class.
            If it is a string, it must be a key in ``TORCH_ACTIVATION``. If None, there will be no activation.
            Defaults to "relu".
        head (bool): If true, this block will be the top block of the full model and the top layer of this block
            will be the final output layer. Defaults to False.
        softmax (bool): If true, the output of the net will be a softmax transformation of the top layer's
            output. Defaults to False.
        batch_norm (bool): If true, batch normalization will be performed at each layer. Defaults to False
        skip_connection (bool): If true, a skip connection will be built between the bottom (input) layer and
            top (output) layer. Defaults to False.
        dropout_p (float): Dropout probability. Defaults to None, in which case there is no drop-out.
        initialize_weights (bool): If true, use the xavier initialization to init the weights of the network. Defaults to False.
        gradient_threshold (float): Gradient clipping threshold. Defaults to None, in which case not gradient clipping
            is performed.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation="relu",
        head: bool = False,
        softmax: bool = False,
        batch_norm: bool = False,
        skip_connection: bool = False,
        dropout_p: float = None,
        gradient_threshold: float = None,
        initialize_weights: bool = False,
        name: str = None
    ):
        super().__init__(
            input_dim, output_dim, hidden_dims,
            activation, head, softmax, batch_norm,
            skip_connection, dropout_p, gradient_threshold, name
        )

        self._net = self._net.double()

        if initialize_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
