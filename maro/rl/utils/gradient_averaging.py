# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch

def average_grad(grad_list: List[str, dict]):
    return {
        param_name: torch.mean(torch.stack([grad[param_name] for grad in grad_list]), dim=0)
        for param_name in grad_list[0]
    }
