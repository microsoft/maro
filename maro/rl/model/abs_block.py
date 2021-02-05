# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn


class AbsBlock(nn.Module):
    @property
    def input_dim(self):
        raise NotImplementedError

    @property
    def output_dim(self):
        raise NotImplementedError
