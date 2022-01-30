# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from maro.rl.distributed import AbsWorker
from maro.rl.utils.common import bytes_to_pyobj, pyobj_to_bytes
from maro.utils import Logger

from .env_sampler import AbsEnvSampler


class RolloutWorker(AbsWorker):
    def __init__(
        self,
        idx: int,
        env_sampler_creator: Callable[[], AbsEnvSampler],
        proxy_host: str,
        proxy_port: int = 10001,
        logger: Logger = None
    ) -> None:
        super(RolloutWorker, self).__init__(idx=idx, proxy_host=proxy_host, proxy_port=proxy_port, logger=logger)
        self._env_sampler = env_sampler_creator()

    def _compute(self, msg: list) -> None:
        if msg[-1] == b"EXIT":
            self._stream.send(b"EXIT_ACK")
            self._logger.info("Exiting event loop...")
            self.stop()
        else:
            req = bytes_to_pyobj(msg[-1])
            assert isinstance(req, dict)
            func = getattr(self._env_sampler, req["func"])
            result = func(*req["args"], **req["kwargs"])
            self._stream.send_multipart([msg[0], pyobj_to_bytes(result)])
