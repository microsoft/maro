# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing

from maro.rl.distributed import AbsWorker
from maro.rl.utils.common import bytes_to_pyobj, pyobj_to_bytes
from maro.utils import LoggerV2

if typing.TYPE_CHECKING:
    from maro.rl.rl_component.rl_component_bundle import RLComponentBundle


class RolloutWorker(AbsWorker):
    """Worker that hosts an environment simulator and executes roll-out on demand for sampling and evaluation purposes.

    Args:
        idx (int): Integer identifier for the worker. It is used to generate an internal ID, "worker.{idx}",
            so that the parallel roll-out controller can keep track of its connection status.
        rl_component_bundle (RLComponentBundle): The RL component bundle of the job.
        producer_host (str): IP address of the parallel task controller host to connect to.
        producer_port (int, default=20000): Port of the parallel task controller host to connect to.
        logger (LoggerV2, default=None): The logger of the workflow.
    """

    def __init__(
        self,
        idx: int,
        rl_component_bundle: RLComponentBundle,
        producer_host: str,
        producer_port: int = 20000,
        logger: LoggerV2 = None,
    ) -> None:
        super(RolloutWorker, self).__init__(
            idx=idx,
            producer_host=producer_host,
            producer_port=producer_port,
            logger=logger,
        )
        self._env_sampler = rl_component_bundle.env_sampler

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
            assert req["type"] in {"sample", "eval", "set_policy_state"}
            if req["type"] == "sample":
                result = self._env_sampler.sample(policy_state=req["policy_state"], num_steps=req["num_steps"])
            elif req["type"] == "eval":
                result = self._env_sampler.eval(policy_state=req["policy_state"])
            else:
                self._env_sampler.set_policy_state(policy_state_dict=req["policy_state"])
                result = True

            self._stream.send(pyobj_to_bytes({"result": result, "index": req["index"]}))
