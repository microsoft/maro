# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np
import scipy.signal


def discount_cumsum(x: Union[np.ndarray, list], discount: float) -> np.ndarray:
    """
    Magic from rllab for computing discounted cumulative sums of vectors.

    Original code from:
        https://github.com/rll/rllab/blob/master/rllab/misc/special.py).

    For details about the scipy function, see:
        https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering

    input:
        vector x,
        [x0, x1, x2]

    output:
        [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
    """
    return np.array(scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1], dtype=np.float32)
