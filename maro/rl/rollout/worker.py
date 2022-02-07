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
        producer_host: str,
        producer_port: int = 20000,
        logger: Logger = None
    ) -> None:
        super(RolloutWorker, self).__init__(idx=idx, producer_host=producer_host, producer_port=producer_port, logger=logger)
        self._env_sampler = env_sampler_creator()

    def _compute(self, msg: list) -> None:
        if msg[-1] == b"EXIT":
            self._logger.info("Exiting event loop...")
            self.stop()
        else:
            req = bytes_to_pyobj(msg[-1])
            assert isinstance(req, dict)
            assert req["type"] in {"sample", "eval"}
            if req["type"] == "sample":
                result = self._env_sampler.sample(policy_state=req["policy_state"], num_steps=req["num_steps"])
            else:
                result = self._env_sampler.eval(policy_state=req["policy_state"])

            self._stream.send(pyobj_to_bytes({"result": result, "index": req["index"]}))
