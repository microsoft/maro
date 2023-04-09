# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import torch

StateType = Any  # State
ObservationType = Any  # Obs
ActionType = Any
PolicyActionType = Union[np.ndarray, torch.Tensor]


@dataclass
class EnvStepRes:
    tick: int
    event: Any
    obs: ObservationType
    agent_obs_dict: Dict[Any, ObservationType]
    end_of_episode: bool

    @classmethod
    def dummy(cls) -> EnvStepRes:
        return EnvStepRes(
            tick=-1,
            event=None,
            obs=None,
            agent_obs_dict={},
            end_of_episode=True,
        )
