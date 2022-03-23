# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys
from typing import Any, Callable, Dict, List

from maro.rl.policy import RLPolicy
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsTrainer


class Scenario(object):
    def __init__(self, path: str) -> None:
        super(Scenario, self).__init__()
        path = os.path.normpath(path)
        sys.path.insert(0, os.path.dirname(path))
        self._module = importlib.import_module(os.path.basename(path))

    @property
    def env_sampler_creator(self) -> AbsEnvSampler:
        return getattr(self._module, "env_sampler_creator")

    @property
    def agent2policy(self) -> Dict[Any, str]:
        return getattr(self._module, "agent2policy")

    @property
    def policy_creator(self) -> Dict[str, Callable[[str], RLPolicy]]:
        return getattr(self._module, "policy_creator")

    @property
    def trainable_policies(self) -> List[str]:
        return getattr(self._module, "trainable_policies", None)

    @property
    def trainer_creator(self) -> Dict[str, Callable[[str], AbsTrainer]]:
        return getattr(self._module, "trainer_creator")

    @property
    def post_collect(self) -> Callable[[list, int, int], None]:
        return getattr(self._module, "post_collect", None)

    @property
    def post_evaluate(self) -> Callable[[list, int], None]:
        return getattr(self._module, "post_evaluate", None)
