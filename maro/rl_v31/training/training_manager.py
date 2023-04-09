# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from collections import defaultdict
from typing import Any, Dict, List

from maro.rl_v31.objects import ExpElement
from maro.rl_v31.policy.base import BaseRLPolicy
from maro.rl_v31.training.trainer import BaseTrainer


class TrainingManager(object):
    def __init__(
        self,
        trainers: List[BaseTrainer],  # TODO: replace by bundle
        policies: List[BaseRLPolicy],  # TODO: replace by bundle
        agent2policy: Dict[Any, str],  # TODO: replace by bundle
        policy2trainer: Dict[str, str],  # TODO: replace by bundle
    ) -> None:

        self._trainer_dict = {}
        for trainer in trainers:
            trainer.register_policies(policies, policy2trainer)
            trainer.register_agent2policy(agent2policy, policy2trainer)
            trainer.build()
            self._trainer_dict[trainer.name] = trainer

        self._policies = policies
        self._agent2policy = agent2policy
        self._policy2trainer = policy2trainer

        self._agent2trainer: Dict[Any, str] = {}
        for agent_name, policy_name in agent2policy.items():
            self._agent2trainer[agent_name] = policy2trainer[policy_name]

    def train_step(self) -> None:
        for trainer in self._trainer_dict.values():
            trainer.train_step()

    def get_policy_state(self) -> Dict[str, dict]:
        policy_states = {}
        for trainer in self._trainer_dict.values():
            policy_states.update(trainer.get_policy_state())
        return policy_states

    def record_exp(self, exps: Dict[int, List[ExpElement]]) -> None:
        for env_id, exp_list in exps.items():
            trainer_exp_pool: Dict[str, List[ExpElement]] = defaultdict(list)
            for element in exp_list:
                exp_dict = element.split_contents_by_trainer(self._agent2trainer)
                for trainer_name, trainer_exp in exp_dict.items():
                    trainer_exp_pool[trainer_name].append(trainer_exp)

            for trainer_name, trainer_exps in trainer_exp_pool.items():
                self._trainer_dict[trainer_name].record_exp(env_id, trainer_exps)

    def load(self, path: str) -> List[str]:
        pass

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for trainer_name, trainer in self._trainer_dict.items():
            trainer.save(path)
