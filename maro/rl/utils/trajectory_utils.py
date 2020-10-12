# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import reduce

import numpy as np


def get_k_step_discounted_sums(arr: np.ndarray, discount: float, k: int = -1):
    if k < 0:
        k = len(arr)
    return reduce(lambda x, y: x * discount + y,
                  [np.pad(arr[i:], (0, i)) for i in range(min(k, len(arr))-1, -1, -1)])


rw = np.asarray([3, 2, 4, 1, 5])
gamma = 0.8

print(get_k_step_discounted_sums(rw, gamma, k=4))


