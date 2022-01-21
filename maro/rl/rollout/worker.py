# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from maro.rl.distributed import AbsWorker
from maro.rl.utils.common import bytes_to_pyobj, pyobj_to_bytes

from .env_sampler import AbsEnvSampler


class RolloutWorker(AbsWorker):
    def __init__(
        self,
        idx: int,
        env_sampler_creator: Callable[[], AbsEnvSampler],
        router_host: str,
        router_port: int = 10001
    ) -> None:
        super(RolloutWorker, self).__init__(
            idx=idx, router_host=router_host, router_port=router_port
        )
        self._env_sampler = env_sampler_creator()

    def _compute(self, msg: list) -> None:
        req = bytes_to_pyobj(msg[-1])
        assert isinstance(req, dict)

        func = getattr(self._env_sampler, req["func"])
        result = func(*req["args"], **req["kwargs"])
        self._receiver.send_multipart([msg[0], pyobj_to_bytes(result)])
