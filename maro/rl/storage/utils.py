# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from functools import wraps

import numpy as np


def check_uniformity(arg_num):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            contents = args[arg_num]
            length = len(contents[next(iter(contents))])
            if any(len(lst) != length for lst in contents.values()):
                raise ValueError("all sequences in contents should have the same length")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def normalize(func):
    @wraps(func)
    def wrapper(size, weights=None, replace=True):
        if weights is not None and not isinstance(weights, np.ndarray):
            weights = np.asarray(weights)

        return func(size, weights / np.sum(weights), replace)

    return wrapper


class OverwriteType(Enum):
    ROLLING = "rolling"
    RANDOM = "random"


def get_update_indexes(size: int, added_size: int, capacity: int, overwrite_type, overwrite_indexes=None):
    if added_size > capacity:
        raise ValueError("size of added items should not exceed the store capacity.")

    num_overwrites = size + added_size - capacity
    if num_overwrites < 0:
        return list(range(size, size + added_size))

    if overwrite_indexes is not None:
        write_indexes = list(range(size, capacity)) + list(overwrite_indexes)
    else:
        # follow the overwrite rule set at init
        if overwrite_type == OverwriteType.ROLLING:
            # using the negative index convention for convenience
            start_index = size - capacity
            write_indexes = list(range(start_index, start_index + added_size))
        else:
            random_indexes = np.random.choice(size, size=num_overwrites, replace=False)
            write_indexes = list(range(size, capacity)) + list(random_indexes)

    return write_indexes
