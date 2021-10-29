from typing import Union

import numpy as np
import torch


def match_shape(tensor: Union[torch.Tensor, np.ndarray], shape: tuple) -> bool:
    """Check if a torch.Tensor/np.ndarray could match the expected shape.

    Args:
        tensor: torch.Tensor or np.ndarray
        shape: The expected shape tuple. If an element in this tuple is None, it means this dimension could match any
            value (usually used for the `batch_size` dimension).

    Returns:
        Whether the tensor could match the expected shape.
    """
    if len(tensor.shape) != len(shape):
        return False
    for val, expected in zip(tensor.shape, shape):
        if expected is not None and expected != val:
            return False
    return True
