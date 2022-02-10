# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from itertools import chain
from typing import Dict, List, Tuple

import zmq
from zmq import Context, Poller

from maro.rl.utils.common import bytes_to_pyobj, get_own_ip_address, pyobj_to_bytes
from maro.utils import DummyLogger, Logger


class ParallelTaskController(object):
    """Controller that sends identical tasks to a set of remote workers and collect results from them.

    Args:
        port (int, default=20000): Network port the controller uses to talk to the remote workers.
        logger (Logger, default=None): Optional logger for logging key events.
    """
    def __init__(self, port: int = 20000, logger: Logger = None) -> None:
        self._ip = get_own_ip_address()
        self._context = Context.instance()

        # parallel task sender
        self._task_endpoint = self._context.socket(zmq.ROUTER)
        self._task_endpoint.setsockopt(zmq.LINGER, 0)
        self._task_endpoint.bind(f"tcp://{self._ip}:{port}")

        self._poller = Poller()
        self._poller.register(self._task_endpoint, zmq.POLLIN)

        self._workers = set()
        self._logger = logger

    def _wait_for_workers_ready(self, k: int):
        while len(self._workers) < k:
            self._workers.add(self._task_endpoint.recv_multipart()[0])

    def _recv_result_for_target_index(self, index: Tuple[int, int]):
        rep = bytes_to_pyobj(self._task_endpoint.recv_multipart()[-1])
        assert isinstance(rep, dict)
        return rep["result"] if rep["index"] == index else None

    def collect(self, req: dict, parallelism: int, min_replies: int = None, grace_factor: int = None) -> List[dict]:
        """Send a task request to a set of remote workers and collect the results.

        Args:
            req (dict): Request containing task specifications and parameters.
            parallelism (int): Number of workers to send the task to.
            min_replies (int, default=None): The minimum number of results to collect in one round of remote
                sampling. If None, it defaults to the value of ``parallelism``.
            grace_factor (float, default=None): Factor that determines the additional wait time after receiving the
                minimum required replies (as determined by ``min_replies``). For example, if the minimum required
                replies are received in T seconds, it will allow an additional T * grace_factor seconds to collect
                the remaining results.

        Returns:
            A list of results. Each element in the list is a dict that contains results from a worker.
        """
        self._wait_for_workers_ready(parallelism)
        if min_replies is None:
            min_replies = parallelism

        start_time = time.time()
        results = []
        for worker_id in list(self._workers)[:parallelism]:
            self._task_endpoint.send_multipart([worker_id, pyobj_to_bytes(req)])
        self._logger.debug(f"Sent {parallelism} roll-out requests...")

        while len(results) < min_replies:
            result = self._recv_result_for_target_index(req["index"])
            if result:
                results.append(result)

        if grace_factor is not None:
            countdown = int((time.time() - start_time) * grace_factor) * 1000  # milliseconds
            self._logger.debug(f"allowing {countdown / 1000} seconds for remaining results")
            while len(results) < parallelism and countdown > 0:
                start = time.time()
                event = dict(self._poller.poll(countdown))
                if self._task_endpoint in event:
                    result = self._recv_result_for_target_index(req["index"])
                    if result:
                        results.append(result)
                countdown -= time.time() - start

        self._logger.debug(f"Received {len(results)} results")
        return results

    def exit(self):
        """Signal the remote workers to exit and terminate the connections.
        """
        for worker_id in self._workers:
            self._task_endpoint.send_multipart([worker_id, b"EXIT"])
        self._task_endpoint.close()
        self._context.term()


class BatchEnvSampler:
    """Facility that samples from multiple copies of an environment in parallel.

    No environment is created here. Instead, it uses a ParallelTaskController to send roll-out requests to a set of
    remote workers and collect results from them.

    Args:
        sampling_parallelism (int): Parallelism for sampling from the environment.
        port (int): Network port that the internal ``ParallelTaskController`` uses to talk to the remote workers.
        min_env_samples (int, default=None): The minimum number of results to collect in one round of remote sampling.
            If it is None, it defaults to the value of ``sampling_parallelism``.
        grace_factor (float, default=None): Factor that determines the additional wait time after receiving the minimum
            required env samples (as determined by ``min_env_samples``). For example, if the minimum required samples
            are received in T seconds, it will allow an additional T * grace_factor seconds to collect the remaining
            results.
        eval_parallelism (int, default=1): Parallelism for policy evaluation on remote workers.
        logger (Logger, default=None): Optional logger for logging key events.
    """
    def __init__(
        self,
        sampling_parallelism: int,
        port: int = 20000,
        min_env_samples: int = None,
        grace_factor: float = None,
        eval_parallelism: int = 1,
        logger: Logger = None
    ) -> None:
        super(BatchEnvSampler, self).__init__()
        self._logger = logger if logger else DummyLogger()
        self._client = ParallelTaskController(port=port, logger=logger)

        self._sampling_parallelism = sampling_parallelism
        self._min_env_samples = min_env_samples if min_env_samples is not None else self._sampling_parallelism
        self._grace_factor = grace_factor
        self._eval_parallelism = eval_parallelism

        self._ep = 0
        self._segment = 0
        self._end_of_episode = True

    def sample(
        self, policy_state: Dict[str, object] = None, num_steps: int = -1
    ) -> dict:
        # increment episode or segment depending on whether the last episode has concluded
        if self._end_of_episode:
            self._ep += 1
            self._segment = 1
        else:
            self._segment += 1

        self._logger.info(f"Collecting roll-out data for episode {self._ep}, segment {self._segment}")
        req = {
            "type": "sample",
            "policy_state": policy_state,
            "num_steps": num_steps,
            "index": (self._ep, self._segment)     
        }
        results = self._client.collect(
            req, self._sampling_parallelism,
            min_replies=self._min_env_samples,
            grace_factor=self._grace_factor
        )
        self._end_of_episode = any(res["end_of_episode"] for res in results)
        merged_experiences = list(chain(*[res["experiences"] for res in results]))  # List[List[ExpElement]]
        return {
            "end_of_episode": self._end_of_episode,
            "experiences": merged_experiences,
            "info": [res["info"][0] for res in results]
        }

    def eval(self, policy_state: Dict[str, object] = None) -> dict:
        req = {"type": "eval", "policy_state": policy_state, "index": (self._ep, -1)}  # -1 signals test
        results = self._client.collect(req, self._eval_parallelism)
        return {
            "info": [res["info"][0] for res in results]
        }

    def exit(self):
        self._client.exit()
