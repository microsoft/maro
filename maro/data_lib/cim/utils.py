# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from random import Random
from typing import List, Union

# we keep 4 random generator to make the result is reproduceable with same seed(s), no matter if agent passed actions
ROUTE_INIT_RAND_KEY = "route_init"
ORDER_INIT_RAND_KEY = "order_init"
BUFFER_TICK_RAND_KEY = "buffer_time"
ORDER_NUM_RAND_KEY = "order_number"

DATA_CONTAINER_INIT_SEED_LIMIT = 4096


def clip(min_val: Union[int, float], max_val: Union[int, float], value: Union[int, float]) -> Union[int, float]:
    """Clip value between specified range

    Args:
        min_val (Union[int, float]): min value to clip
        max_val (Union[int, float]): max value to clip
        value (Union[int, float]): value to clip

    Returns:
        Union[int, float]: clipped value
    """
    return max(min_val, min(max_val, value))


def apply_noise(value: Union[int, float], noise: Union[int, float], rand: Random) -> float:
    """Apply noise with specified random generator

    Args:
        value (int): number to apply noise
        noise (int): noise range
        rand (SimRandom): random generator

    Returns:
        float: final value after apply noise
    """
    return value + rand.uniform(-noise, noise)


def list_sum_normalize(num_list: List[Union[int, float]]) -> List[float]:
    """Normalize with sum.

    Args:
        num_list(list): List of number to normalize.

    Returns:
        list: List of normalized number.
    """
    t = sum(num_list)

    # avoid dive zero exception
    if t == 0:
        return 0

    return [d / t for d in num_list]
