# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from maro.rl.distributed import AbsWorker
from maro.rl.utils.common import bytes_to_pyobj, pyobj_to_bytes
from maro.utils import Logger

from .env_sampler import AbsEnvSampler


class RolloutWorker(AbsWorker):
    """Worker that hosts an environment simulator and executes roll-out on demand for sampling and evaluation purposes.

    Args:
        idx (int): Integer identifier for the worker. It is used to generate an internal ID, "worker.{idx}",
            so that the parallel roll-out controller can keep track of its connection status.
        env_sampler_creator (Callable[[dict], AbsEnvSampler]): User-defined function to create an ``AbsEnvSampler``
            for roll-out purposes. 
        producer_host (str): IP address of the parallel task controller host to connect to.
        producer_port (int, default=10001): Port of the parallel task controller host to connect to.
    """
    def __init__(
        self,
        idx: int,
        env_sampler_creator: Callable[[], AbsEnvSampler],
        producer_host: str,
        producer_port: int = 20000,
        logger: Logger = None
    ) -> None:
        super(RolloutWorker, self).__init__(
            idx=idx, producer_host=producer_host, producer_port=producer_port, logger=logger
        )
        self._env_sampler = env_sampler_creator()

    def _compute(self, msg: list) -> None:
        """Perform a full or partial episode of roll-out for sampling or evaluation.

        Args:
            msg (list): Multi-part message containing roll-out specifications and parameters.
        """
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
