from typing import Union

import numpy as np
import torch

from .objects import SHAPE_CHECK_FLAG


def match_shape(tensor: Union[torch.Tensor, np.ndarray], shape: tuple) -> bool:
    """Check if a torch.Tensor/np.ndarray could match the expected shape.

    Args:
        tensor: torch.Tensor or np.ndarray
        shape: The expected shape tuple. If an element in this tuple is None, it means this dimension could match any
            value (usually used for the `batch_size` dimension).

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


def ndarray_to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a np.ndarray to a torch.Tensor.

    Args:
        array (np.ndarray): The input ndarray.
        device (torch.device): The device to assign this tensor.

    Returns:
        A tensor with same shape and values.
    """
    return torch.from_numpy(array).to(device)
