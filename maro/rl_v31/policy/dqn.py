# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any, Optional

import torch
from gym import spaces
from tianshou.data import Batch
from torch import nn
from torch.optim import Optimizer

from maro.rl_v31.exploration.strategy import ExploreStrategy
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
        explore_strategy: Optional[ExploreStrategy] = None,
    ) -> None:
        super().__init__(
            name=name,
            obs_space=obs_space,
            action_space=action_space,
            optim=optim,
        )

        self._qnet = qnet
        self._explore_strategy = explore_strategy

    def q_values(self, obs: Any, act: torch.Tensor) -> torch.Tensor:
        return self._qnet.q_values(obs, act)

    def q_values_for_all(self, obs: Any) -> torch.Tensor:
        return self._qnet.q_values_for_all(obs)

    def forward(
        self,
        batch: Batch,
        **kwargs: Any,
    ) -> Batch:
        obs = to_torch(batch.obs)
        q = self.q_values_for_all(obs)
        q_softmax = nn.Softmax(dim=1)(q)
        _, act = q.max(dim=1)

        if self.is_exploring and self._explore_strategy is not None:
            act = self._explore_strategy.get_action(obs=obs.cpu().numpy(), action=act.cpu().numpy())
            act = to_torch(act)

        act = act.unsqueeze(1).long()
        probs = q_softmax.gather(1, act).squeeze(-1)
        return Batch(act=act, probs=probs)
