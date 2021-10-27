# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch
import torch.nn as nn

class FCN_model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation=nn.ReLU,
        dropout_p: float = 0,
    ):
        super().__init__()

        layers = [
            nn.Linear(in_features=input_dim, out_features=hidden_dims[0]),
            activation()
        ]

        for in_dim, out_dim in zip(hidden_dims, hidden_dims[1:]):
            layers.extend(
                [
                    nn.Dropout(p=dropout_p),
                    nn.Linear(in_features=in_dim, out_features=out_dim),
                    nn.BatchNorm1d(num_features=out_dim),
                    activation()
                ]
            )

        layers.append(
            nn.Linear(in_features=hidden_dims[-1], out_features=output_dim)
        )

        self._net = nn.Sequential(*layers)
        print(self._net)

    def forward(self, x):
        return self._net(x)

"""
Sequential(
  (0): Linear(in_features=3, out_features=128, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.15, inplace=False)
  (3): Linear(in_features=128, out_features=128, bias=True)
  (4): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU()
  (6): Dropout(p=0.15, inplace=False)
  (7): Linear(in_features=128, out_features=128, bias=True)
  (8): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (9): ReLU()
  (10): Linear(in_features=128, out_features=3, bias=True)
)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 3)]               0
_________________________________________________________________
dense (Dense)                (None, 128)               512
_________________________________________________________________
dropout (Dropout)            (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512
_________________________________________________________________
batch_normalization (BatchNo (None, 128)               512
_________________________________________________________________
activation (Activation)      (None, 128)               0
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_2 (Dense)              (None, 128)               16512
_________________________________________________________________
batch_normalization_1 (Batch (None, 128)               512
_________________________________________________________________
activation_1 (Activation)    (None, 128)               0
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 387
=================================================================
"""
