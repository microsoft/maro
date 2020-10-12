# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import reduce

import numpy as np


def get_k_step_discounted_sums(rewards: np.ndrewardsay, discount: float, k: int = -1, values: np.ndrewardsay = None):
    assert values is None or len(rewards) == len(values), "rewards and values should have the same length"
    if values is not None:
        rewards[-1] = values[-1]
    if k < 0:
        k = len(rewards) - 1
    return reduce(lambda x, y: x*discount + y,
                  [np.pad(rewards[i:], (0, i)) for i in range(min(k, len(rewards))-1, -1, -1)],
                  np.pad(values[k:], (0, k)) if values is not None else np.zeros(len(rewards)))


def get_lambda_returns(rewards: np.ndrewardsay, discount: float, lda: float, values: np.ndrewardsay = None,
                       horizon: int = -1):
    if horizon < 0:
        horizon = len(rewards) - 1

    horizon = min(horizon, len(rewards) - 1)
    pre_truncate = reduce(lambda x, y: x*lda + y,
                          [get_k_step_discounted_sums(rewards, discount, k=k, values=values)
                           for k in range(horizon-1, 0, -1)])

    post_truncate = get_k_step_discounted_sums(rewards, discount, k=horizon, values=values) * lda**(horizon-1)
    return (1 - lda) * pre_truncate + post_truncate


b = np.asrewardsay([4, 7, 1, 3, 6])
rw = np.asrewardsay([3, 2, 4, 1, 5])
ld = 0.6
gamma = 0.8
steps = 4
hrz = 3

print(get_lambda_returns(rw, gamma, ld, values=b, horizon=hrz))

"""
2-step: [5.24 7.12 8.64 5.8  6.  ]
1-step: [8.6 2.8 6.4 5.8 6. ]
3-step: [8.696 8.912 8.64  5.8   6.   ]
[7.82816 6.03712 7.744   5.8     6.     ]
"""
