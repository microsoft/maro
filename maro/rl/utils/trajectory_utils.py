# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import reduce

import numpy as np


def get_k_step_returns(rewards: np.ndarray, discount: float, k: int = -1, values: np.ndarray = None):
    """Compute K-step returns given reward and value sequences.
    Args:
        rewards (np.ndarray): reward sequence from a trajectory.
        discount (float): reward discount as in standard RL.
        k (int): number of steps in computing returns. If it is -1, returns are computed using the largest possible
            number of steps. Defaults to -1.
        values (np.ndarray): sequence of values for the traversed states in a trajectory. If it is None, the state
            immediately after the final state in the given sequence is assumed to be terminal with value zero, and the
            computed returns for k = -1 are actual full returns. Defaults to None.

    Returns:
        An ndarray containing the k-step returns for each time step.
    """
    assert values is None or len(rewards) == len(values), "rewards and values should have the same length"
    if values is not None:
        rewards[-1] = values[-1]
    if k < 0:
        k = len(rewards) - 1
    return reduce(lambda x, y: x*discount + y,
                  [np.pad(rewards[i:], (0, i)) for i in range(min(k, len(rewards))-1, -1, -1)],
                  np.pad(values[k:], (0, k)) if values is not None else np.zeros(len(rewards)))


def get_lambda_returns(rewards: np.ndarray, discount: float, lmda: float, values: np.ndarray = None,
                       truncate_steps: int = -1):
    """Compute lambda returns given reward and value sequences and a truncate_steps.
    Args:
        rewards (np.ndarray): reward sequence from a trajectory.
        discount (float): reward discount as in standard RL.
        lmda (float): the lambda coefficient involved in computing lambda returns.
        values (np.ndarray): sequence of values for the traversed states in a trajectory. If it is None, the state
            immediately after the final state in the given sequence is assumed to be terminal with value zero.
            Defaults to None.
        truncate_steps (int): number of steps where the lambda return series is truncated. If it is -1, no truncating
            is done and the lambda return is carried out to the end of the sequence. Defaults to -1.

    Returns:
        An ndarray containing the lambda returns for each time step.
    """
    if truncate_steps < 0:
        truncate_steps = len(rewards) - 1

    truncate_steps = min(truncate_steps, len(rewards) - 1)
    pre_truncate = reduce(lambda x, y: x*lmda + y,
                          [get_k_step_returns(rewards, discount, k=k, values=values)
                           for k in range(truncate_steps-1, 0, -1)])

    post_truncate = get_k_step_returns(rewards, discount, k=truncate_steps, values=values) * lmda**(truncate_steps-1)
    return (1 - lmda) * pre_truncate + post_truncate


b = np.asarray([4, 7, 1, 3, 6])
rw = np.asarray([3, 2, 4, 1, 5])
ld = 0.6
gamma = 0.8
steps = 4
hrz = 3

print(get_lambda_returns(rw, gamma, ld, values=None, truncate_steps=3))

"""
2-step: [5.24 7.12 8.64 5.8  6.  ]
1-step: [8.6 2.8 6.4 5.8 6. ]
3-step: [8.696 8.912 8.64  5.8   6.   ]
[7.82816 6.03712 7.744   5.8     6.     ]
"""
