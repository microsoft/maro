# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any, Optional, Union

import gym
import numpy as np
import scipy.signal
import torch
from gym import spaces


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


class GymSpaceValidationError(Exception):
    def __init__(self, message: str, space: gym.Space, x: Any) -> None:
        self.message = message
        self.space = space
        self.x = x

    def __str__(self) -> str:
        return f"{self.message}\n  Space: {self.space}\n  Sample: {self.x}"


def gym_space_contains(space: gym.Space, x: Any) -> None:
    """Strengthened version of gym.Space.contains.
    Giving more diagnostic information on why validation fails.
    Throw exception rather than returning true or false.

    From https://github.com/microsoft/qlib/blob/main/qlib/rl/interpreter.py.
    """
    if isinstance(space, spaces.Dict):
        if not isinstance(x, dict) or len(x) != len(space):
            raise GymSpaceValidationError("Sample must be a dict with same length as space.", space, x)
        for k, subspace in space.spaces.items():
            if k not in x:
                raise GymSpaceValidationError(f"Key {k} not found in sample.", space, x)
            try:
                gym_space_contains(subspace, x[k])
            except GymSpaceValidationError as e:
                raise GymSpaceValidationError(f"Subspace of key {k} validation error.", space, x) from e

    elif isinstance(space, spaces.Tuple):
        if isinstance(x, (list, np.ndarray)):
            x = tuple(x)  # Promote list and ndarray to tuple for contains check
        if not isinstance(x, tuple) or len(x) != len(space):
            raise GymSpaceValidationError("Sample must be a tuple with same length as space.", space, x)
        for i, (subspace, part) in enumerate(zip(space, x)):
            try:
                gym_space_contains(subspace, part)
            except GymSpaceValidationError as e:
                raise GymSpaceValidationError(f"Subspace of index {i} validation error.", space, x) from e

    else:
        if not space.contains(x):
            raise GymSpaceValidationError("Validation error reported by gym.", space, x)


def convert_ndarray_to_tensor(obs: Any) -> Any:
    if isinstance(obs, np.ndarray):
        return torch.from_numpy(obs)
    elif isinstance(obs, list):
        return [convert_ndarray_to_tensor(e) for e in obs]
    elif isinstance(obs, tuple):
        return tuple([convert_ndarray_to_tensor(e) for e in obs])
    elif isinstance(obs, dict):
        return {k: convert_ndarray_to_tensor(v) for k, v in obs.items()}
    else:
        return obs


def align_device(obs: Any, device: Optional[torch.device] = None) -> None:
    if isinstance(obs, torch.Tensor):
        obs.to(device)
    elif isinstance(obs, (list, tuple)):
        for sub_obs in obs:
            align_device(sub_obs)
    elif isinstance(obs, dict):
        for v in obs.values():
            align_device(v)
    else:
        pass  # Ignore non tensor objects


def to_torch(obs: Any, device: Optional[torch.device] = None) -> Any:
    # TODO: handle device
    obs = convert_ndarray_to_tensor(obs)
    align_device(obs, device)
    return obs
