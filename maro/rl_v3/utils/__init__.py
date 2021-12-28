from typing import Union

from .objects import SHAPE_CHECK_FLAG
from .torch_util import match_shape, ndarray_to_tensor
from .transition_batch import MultiTransitionBatch, TransitionBatch


AbsTransitionBatch = Union[TransitionBatch, MultiTransitionBatch]


__all__ = [
    "SHAPE_CHECK_FLAG",
    "match_shape", "ndarray_to_tensor",
    "AbsTransitionBatch", "MultiTransitionBatch", "TransitionBatch"
]
