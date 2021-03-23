# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import reduce
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


def get_truncated_cumulative_reward(
    rewards: Union[list, np.ndarray, torch.Tensor],
    discount: float,
    k: int = -1
):
    """Compute K-step cumulative rewards from a reward sequence.
    Args:
        rewards (Union[list, np.ndarray, torch.Tensor]): Reward sequence from a trajectory.
        discount (float): Reward discount as in standard RL.
        k (int): Number of steps in computing cumulative rewards. If it is -1, returns are computed using the
            largest possible number of steps. Defaults to -1.

    Returns:
        An ndarray or torch.Tensor instance containing the k-step cumulative rewards for each time step.
    """
    if k < 0:
        k = len(rewards) - 1
    pad = np.pad if isinstance(rewards, list) or isinstance(rewards, np.ndarray) else F.pad
    return reduce(
        lambda x, y: x * discount + y,
        [pad(rewards[i:], (0, i)) for i in range(min(k, len(rewards)) - 1, -1, -1)]
    )


def get_k_step_returns(
    rewards: Union[list, np.ndarray, torch.Tensor],
    values: Union[list, np.ndarray, torch.Tensor],
    discount: float,
    k: int = -1
):
    """Compute K-step returns given reward and value sequences.
    Args:
        rewards (Union[list, np.ndarray, torch.Tensor]): Reward sequence from a trajectory.
        values (Union[list, np.ndarray, torch.Tensor]): Sequence of values for the traversed states in a trajectory.
        discount (float): Reward discount as in standard RL.
        k (int): Number of steps in computing returns. If it is -1, returns are computed using the largest possible
            number of steps. Defaults to -1.

    Returns:
        An ndarray or torch.Tensor instance containing the k-step returns for each time step.
    """
    assert len(rewards) == len(values), "rewards and values should have the same length"
    assert len(values.shape) == 1, "values should be a one-dimensional array"
    rewards[-1] = values[-1]
    if k < 0:
        k = len(rewards) - 1
    pad = np.pad if isinstance(rewards, list) or isinstance(rewards, np.ndarray) else F.pad
    return reduce(
        lambda x, y: x * discount + y,
        [pad(rewards[i:], (0, i)) for i in range(min(k, len(rewards)) - 1, -1, -1)],
        pad(values[k:], (0, k))
    )


def get_lambda_returns(
    rewards: Union[list, np.ndarray, torch.Tensor],
    values: Union[list, np.ndarray, torch.Tensor],
    discount: float,
    lam: float,
    k: int = -1
):
    """Compute lambda returns given reward and value sequences and a k.
    Args:
        rewards (Union[list, np.ndarray, torch.Tensor]): Reward sequence from a trajectory.
        values (Union[list, np.ndarray, torch.Tensor]): Sequence of values for the traversed states in a trajectory.
        discount (float): Reward discount as in standard RL.
        lam (float): Lambda coefficient involved in computing lambda returns.
        k (int): Number of steps where the lambda return series is truncated. If it is -1, no truncating is done and
            the lambda return is carried out to the end of the sequence. Defaults to -1.

    Returns:
        An ndarray or torch.Tensor instance containing the lambda returns for each time step.
    """
    if k < 0:
        k = len(rewards) - 1

    # If lambda is zero, lambda return reduces to one-step return
    if lam == .0:
        return get_k_step_returns(rewards, values, discount, k=1)

    # If lambda is one, lambda return reduces to k-step return
    if lam == 1.0:
        return get_k_step_returns(rewards, values, discount, k=k)

    k = min(k, len(rewards) - 1)
    pre_truncate = reduce(
        lambda x, y: x * lam + y,
        [get_k_step_returns(rewards, values, discount, k=k) for k in range(k - 1, 0, -1)]
    )

    post_truncate = get_k_step_returns(rewards, values, discount, k=k) * lam**(k - 1)
    return (1 - lam) * pre_truncate + post_truncate
