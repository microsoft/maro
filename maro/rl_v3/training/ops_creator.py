# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Callable, Dict

from maro.rl_v3.policy import RLPolicy

from .trainer import AbsTrainer


class OpsCreator(object):
    def __init__(
        self,
        policy_creator: Dict[str, Callable[[str], RLPolicy]],
        trainer_creator: Dict[str, Callable[[str], AbsTrainer]]
    ) -> None:
        self._policy_creator = policy_creator
        self._trainer_creator = trainer_creator
        self._trainer_dict = {}

    def __getitem__(self, ops_name: str):
        trainer_name = ops_name.split(".")[0]
        if trainer_name not in self._trainer_dict:
            trainer = self._trainer_creator[trainer_name](trainer_name)
            trainer.register_policy_creator(self._policy_creator)
            self._trainer_dict[trainer_name] = trainer
        return partial(self._trainer_dict[trainer_name].get_local_ops_by_name, ops_name)
