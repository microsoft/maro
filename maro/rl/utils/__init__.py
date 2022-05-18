# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

from .objects import SHAPE_CHECK_FLAG
from .torch_utils import average_grads, get_torch_device, match_shape, ndarray_to_tensor
from .trajectory_computation import discount_cumsum
from .transition_batch import MultiTransitionBatch, TransitionBatch, merge_transition_batches

AbsTransitionBatch = Union[TransitionBatch, MultiTransitionBatch]

__all__ = [
    "SHAPE_CHECK_FLAG",
    "average_grads",
    "get_torch_device",
    "match_shape",
    "ndarray_to_tensor",
    "discount_cumsum",
    "AbsTransitionBatch",
    "MultiTransitionBatch",
    "TransitionBatch",
    "merge_transition_batches",
]
