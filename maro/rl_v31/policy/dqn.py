# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any

from gym import spaces
from tianshou.data import Batch
from torch import nn
from torch.optim import Optimizer

from maro.rl_v31.model.qnet import DiscreteQNet
from maro.rl_v31.policy.base import BaseRLPolicy
from maro.rl_v31.utils import to_torch


class DQNPolicy(BaseRLPolicy):
    def __init__(
        self,
        name: str,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        optim: Optimizer,
        qnet: DiscreteQNet,
    ) -> None:
        super().__init__(
            name=name,
            obs_space=obs_space,
            action_space=action_space,
            optim=optim,
        )

        self._qnet = qnet

    def forward(
        self,
        batch: Batch,
        **kwargs: Any,
    ) -> Batch:
        obs = to_torch(batch.obs)
        q = self._qnet.q_values_for_all(obs)
        q_softmax = nn.Softmax(dim=1)(q)
        _, act = q.max(dim=1)

        act = act.unsqueeze(1)
        probs = q_softmax.gather(1, act).squeeze(-1)
        return Batch(act=act, probs=probs)
