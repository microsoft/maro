# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys
from typing import Callable, Dict

from maro.rl.policy import RLPolicy
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsAlgorithm


class Scenario(object):
    def __init__(self, path: str) -> None:
        super(Scenario, self).__init__()
        path = os.path.normpath(path)
        sys.path.insert(0, os.path.dirname(path))
        self._module = importlib.import_module(os.path.basename(path))

    def get_env_sampler(self, policy_creator: Dict[str, Callable[[str], RLPolicy]]) -> AbsEnvSampler:
        return getattr(self._module, "env_sampler_creator")(policy_creator)

    @property
    def agent2policy(self) -> Dict[str, str]:
        return getattr(self._module, "agent2policy")

    @property
    def policy_creator(self) -> Dict[str, Callable[[str], RLPolicy]]:
        return getattr(self._module, "policy_creator")

    @property
    def algorithm_instance_creator(self) -> Dict[str, Callable[[str], AbsAlgorithm]]:
        return getattr(self._module, "algorithm_instance_creator")

    @property
    def post_collect(self) -> Callable[[list, int, int], None]:
        return getattr(self._module, "post_collect", None)

    @property
    def post_evaluate(self) -> Callable[[list, int], None]:
        return getattr(self._module, "post_evaluate", None)
