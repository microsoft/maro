# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

from maro.simulator.utils.sim_random import SimRandom, random

# we keep 4 random generator to make the result is reproduceable with same seed(s), no matter if agent passed actions
route_init_rand = random["route_init"]
order_init_rand = random["order_init"]
buffer_tick_rand = random["buffer_time"]
order_num_rand = random["order_number"]


def get_buffer_tick_seed():
    return random.get_seed("buffer_time")


def get_order_num_seed():
    return random.get_seed("order_number")


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


def apply_noise(value: Union[int, float], noise: Union[int, float], rand: SimRandom) -> float:
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
