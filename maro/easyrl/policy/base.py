# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta
from typing import Dict, List

import numpy as np

from maro.rl.policy import RLPolicy
from maro.rl.rollout import ExpElement
from maro.rl.training import SingleAgentTrainer


class EasyPolicy(object, metaclass=ABCMeta):
    def __init__(
        self,
        actor: RLPolicy,
        trainer: SingleAgentTrainer,
    ) -> None:
        self._actor = actor
        self._trainer = trainer

        self._trainer.register_policy_creator(
            global_policy_creator={actor.name: lambda: actor},
            policy_trainer_mapping={actor.name: trainer.name},
        )
        self._trainer.build()

    @property
    def actor(self) -> RLPolicy:
        return self._actor

    @property
    def name(self) -> str:
        return self._actor.name

    def get_actions(self, states: np.ndarray) -> np.ndarray:
        return self._actor.get_actions(states)

    def train_step(self) -> None:
        self._trainer.train_step()

    def record_experiences(self, experiences: Dict[int, List[ExpElement]]) -> None:
        # TODO: support multi agent
        for env_idx, exps in experiences.items():
            self._trainer.record_multiple(env_idx, exps)

    def train(self, mode: bool = True) -> None:
        if mode:
            self._actor.train()
        else:
            self._actor.eval()
