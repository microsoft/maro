# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import numpy as np
import torch

from .objects import SHAPE_CHECK_FLAG


def match_shape(tensor: Union[torch.Tensor, np.ndarray], shape: tuple) -> bool:
    """Check if a torch.Tensor / np.ndarray could match the expected shape.

    Args:
        tensor (Union[torch.Tensor, np.ndarray]): Tensor.
        shape (tuple): The expected shape tuple. If an element in this tuple is None, it means this dimension
            could match any value (usually used for the `batch_size` dimension).

    Returns:
        Whether the tensor could match the expected shape.
    """
    if not SHAPE_CHECK_FLAG:
        return True
    else:
        if len(tensor.shape) != len(shape):
            return False
        for val, expected in zip(tensor.shape, shape):
            if expected is not None and expected != val:
                return False
        return True


def ndarray_to_tensor(array: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """
    Convert a np.ndarray to a torch.Tensor.

    Args:
        array (np.ndarray): The input ndarray.
        device (torch.device): The device to assign this tensor.

    Returns:
        A tensor with same shape and values.
    """
    return torch.from_numpy(array).to(device)


def average_grads(grad_list: List[dict]) -> dict:
    """Obtain the average of a list of gradients."""
    if len(grad_list) == 1:
        return grad_list[0]
    return {
        param_name: torch.mean(torch.stack([grad[param_name] for grad in grad_list]), dim=0)
        for param_name in grad_list[0]
    }


def get_torch_device(device: str = None) -> torch.device:
    return torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
