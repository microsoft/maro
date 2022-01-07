# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import time
from asyncio.tasks import FIRST_COMPLETED
from random import choices
from typing import Dict, List, Tuple

from maro.utils import DummyLogger, Logger
from maro.rl_v3.distributed import RemoteObj

from .env_sampler import ExpElement


class BatchEnvSampler:
    def __init__(
        self,
        num_samplers: int,
        dispatcher_address: Tuple[str, int],
        num_steps: int = -1,
        min_env_samples: int = None,
        collect_time_watermark: float = None,
        num_eval_samples: int = 1,
        prefix: str = "env_sampler", 
        logger: Logger = DummyLogger()
    ) -> None:
        if num_eval_samples > num_samplers:
            raise ValueError("num_eval_workers cannot exceed the number of available workers")

        super(BatchEnvSampler, self).__init__()
        self._logger = logger
        self._remote_samplers = [RemoteObj(f"{prefix}.{i}", dispatcher_address) for i in range(num_samplers)]
        self._num_steps = num_steps
        self._min_env_samples = min_env_samples
        self._collect_time_watermark = collect_time_watermark
        self._num_eval_samples = num_eval_samples

    async def collect(
        self, ep: int, segment: int, policy_state: Dict[str, object]
    ) -> Tuple[List[List[ExpElement]], List[dict]]:
        self._logger.info(f"Collecting simulation data (episode {ep}, segment {segment})")
        if self._min_env_samples is None:
            results = await asyncio.wait(
                *[sampler.sample(policy_state, num_steps=self._num_steps) for sampler in self._remote_samplers]
            )
        else:
            start_time = time.time()
            results = []
            pending = {
                asyncio.create_task(sampler.sample(policy_state, num_steps=self._num_steps))
                for sampler in self._remote_samplers
            }
            while len(results) < self._min_env_samples:
                cur_done, pending = await asyncio.wait(pending, return_when=FIRST_COMPLETED)
                results.extend([task.result() for task in cur_done])

            if self._collect_time_watermark is not None:
                extra_wait_time = (time.time() - start_time) * self._collect_time_watermark
                extra_done, pending = await asyncio.wait(pending, timeout=extra_wait_time)
                results.extend([task.result() for task in extra_done])
                for task in pending:
                    task.cancel()

        self._end_of_episode = any(res["end_of_episode"] for res in results)
        return [res["experiences"] for res in results], [res["tracker"] for res in results]

    async def evaluate(self, policy_state: Dict[str, object]) -> List[dict]:
        samplers = choices(self._remote_samplers, k=self._num_eval_samples)
        results = await asyncio.wait(*[sampler.test(policy_state) for sampler in samplers])
        return [res["tracker"] for res in results]
