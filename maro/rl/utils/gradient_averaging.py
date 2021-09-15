# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch


def average_grads(grad_list: List[dict]):
    """Obtain the average of a list of gradients."""
    return {
        param_name: torch.mean(torch.stack([grad[param_name] for grad in grad_list]), dim=0)
        for param_name in grad_list[0]
    }
