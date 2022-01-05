# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from maro.rl_v3.distributed import AbsWorker
from maro.rl_v3.rollout import AbsEnvSampler


class RolloutWorker(AbsWorker):
    def __init__(
        self,
        idx: int,
        get_env_sampler_func: Callable[[], AbsEnvSampler],
        router_host: str,
        router_port: int = 10001
    ) -> None:
        super(RolloutWorker, self).__init__(idx, router_host, router_port=router_port)
        self._get_env_sampler_func = get_env_sampler_func

    def _create_obj(self, name: str):
        return self._get_env_sampler_func()
