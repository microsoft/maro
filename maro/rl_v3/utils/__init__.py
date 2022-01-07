from typing import Union

from .common import CoroutineWrapper
from .objects import SHAPE_CHECK_FLAG
from .torch_util import match_shape, ndarray_to_tensor
from .trajectory_computation import discount_cumsum
from .transition_batch import MultiTransitionBatch, TransitionBatch

AbsTransitionBatch = Union[TransitionBatch, MultiTransitionBatch]


__all__ = [
    "CoroutineWrapper",
    "SHAPE_CHECK_FLAG",
    "match_shape", "ndarray_to_tensor",
    "discount_cumsum",
    "AbsTransitionBatch", "MultiTransitionBatch", "TransitionBatch",
]
