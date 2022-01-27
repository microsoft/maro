# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from maro.rl.distributed import AbsWorker
from maro.rl.utils.common import bytes_to_pyobj, pyobj_to_bytes
from maro.utils import Logger

from .env_sampler import AbsEnvSampler


class RolloutWorker(AbsWorker):
    """Rollout worker that used to hold a mirror of the environment and run the interactive rollout.

    Args:
        idx (int): Index of this rollout worker.
        env_sampler_creator (Callable[[], AbsEnvSampler]): Function used to create the mirror of the environment.
        router_host (str): Host of the rollout router.
        router_port (int, default=10001): Port of the rollout router.
    """
    def __init__(
        self,
        idx: int,
        env_sampler_creator: Callable[[], AbsEnvSampler],
        router_host: str,
        router_port: int = 10001,
        logger: Logger = None
    ) -> None:
        super(RolloutWorker, self).__init__(
            idx=idx, router_host=router_host, router_port=router_port, logger=logger
        )
        self._env_sampler = env_sampler_creator()

    def _compute(self, msg: list) -> None:
        """Forward the request to the environment and then send the results back to the requester.

        Args:
            msg (list): Message list.
        """
        req = bytes_to_pyobj(msg[-1])
        assert isinstance(req, dict)

        func = getattr(self._env_sampler, req["func"])
        result = func(*req["args"], **req["kwargs"])
        self._receiver.send_multipart([msg[0], pyobj_to_bytes(result)])
