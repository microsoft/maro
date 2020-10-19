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
    if values is not None:
        assert len(rewards) == len(values), "rewards and values should have the same length"
        assert len(values.shape) == 1, "values should be a one-dimensional array"
        rewards[-1] = values[-1]
    if k < 0:
        k = len(rewards) - 1
    return reduce(lambda x, y: x*discount + y,
                  [np.pad(rewards[i:], (0, i)) for i in range(min(k, len(rewards))-1, -1, -1)],
                  np.pad(values[k:], (0, k)) if values is not None else np.zeros(len(rewards)))


def get_lambda_returns(rewards: np.ndarray, discount: float, lam: float, k: int = -1, values: np.ndarray = None):
    """Compute lambda returns given reward and value sequences and a k.
    Args:
        rewards (np.ndarray): reward sequence from a trajectory.
        discount (float): reward discount as in standard RL.
        lam (float): the lambda coefficient involved in computing lambda returns.
        k (int): number of steps where the lambda return series is truncated. If it is -1, no truncating is done and
            the lambda return is carried out to the end of the sequence. Defaults to -1.
        values (np.ndarray): sequence of values for the traversed states in a trajectory. If it is None, the state
            immediately after the final state in the given sequence is assumed to be terminal with value zero.
            Defaults to None.

    Returns:
        An ndarray containing the lambda returns for each time step.
    """
    if k < 0:
        k = len(rewards) - 1

    # If lambda is zero, lambda return reduces to one-step return
    if lam == .0:
        return get_k_step_returns(rewards, discount, k=1, values=values)

    # If lambda is one, lambda return reduces to k-step return
    if lam == 1.0:
        return get_k_step_returns(rewards, discount, k=k, values=values)

    k = min(k, len(rewards) - 1)
    pre_truncate = reduce(lambda x, y: x*lam + y,
                          [get_k_step_returns(rewards, discount, k=k, values=values)
                           for k in range(k-1, 0, -1)])

    post_truncate = get_k_step_returns(rewards, discount, k=k, values=values) * lam**(k-1)
    return (1 - lam) * pre_truncate + post_truncate
