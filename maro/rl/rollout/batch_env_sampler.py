# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
from itertools import chain
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import zmq
from zmq import Context, Poller

from maro.rl.distributed import DEFAULT_ROLLOUT_PRODUCER_PORT
from maro.rl.utils.common import bytes_to_pyobj, get_own_ip_address, pyobj_to_bytes
from maro.rl.utils.objects import FILE_SUFFIX
from maro.utils import DummyLogger, LoggerV2

from .env_sampler import EnvSamplerInterface, ExpElement


def _split(total: int, k: int) -> List[int]:
    """Split integer `total` into `k` groups where the sum of the `k` groups equals to `total` and all groups are
    as close as possible.
    """

    p, q = total // k, total % k
    return [p + 1] * q + [p] * (k - q)


class ParallelTaskController(object):
    """Controller that sends identical tasks to a set of remote workers and collect results from them.

    Args:
        port (int, default=20000): Network port the controller uses to talk to the remote workers.
        logger (LoggerV2, default=None): Optional logger for logging key events.
    """

    def __init__(self, port: int = 20000, logger: LoggerV2 = None) -> None:
        self._ip = get_own_ip_address()
        self._context = Context.instance()

        # parallel task sender
        self._task_endpoint = self._context.socket(zmq.ROUTER)
        self._task_endpoint.setsockopt(zmq.LINGER, 0)
        self._task_endpoint.bind(f"tcp://{self._ip}:{port}")

        self._poller = Poller()
        self._poller.register(self._task_endpoint, zmq.POLLIN)

        self._workers: set = set()
        self._logger: Union[DummyLogger, LoggerV2] = logger or DummyLogger()

    def _wait_for_workers_ready(self, k: int) -> None:
        while len(self._workers) < k:
            self._workers.add(self._task_endpoint.recv_multipart()[0])

    def _recv_result_for_target_index(self, index: int) -> Any:
        rep = bytes_to_pyobj(self._task_endpoint.recv_multipart()[-1])
        assert isinstance(rep, dict)
        return rep["result"] if rep["index"] == index else None

    def collect(
        self,
        req: dict,
        parallelism: int,
        min_replies: int = None,
        grace_factor: float = None,
        unique_params: Optional[dict] = None,
    ) -> list:
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
            unique_params (Optional[float], default=None): Unique params for each worker.

        Returns:
            A list of results. Each element in the list is a dict that contains results from a worker.
        """
        if unique_params is not None:
            for key, params in unique_params.items():
                assert len(params) == parallelism
            assert len(set(req.keys()) & set(unique_params.keys())) == 0, "Parameter overwritten is not allowed."

        self._wait_for_workers_ready(parallelism)
        min_replies = min_replies or parallelism

        start_time = time.time()
        results: list = []
        for i, worker_id in enumerate(list(self._workers)[:parallelism]):
            cur_params = {key: params[i] for key, params in unique_params.items()} if unique_params is not None else {}
            self._task_endpoint.send_multipart([worker_id, pyobj_to_bytes({**req, **cur_params})])
        self._logger.debug(f"Sent {parallelism} roll-out requests...")

        while len(results) < min_replies:
            result = self._recv_result_for_target_index(req["index"])
            if result:
                results.append(result)

        if grace_factor is not None:
            countdown = int((time.time() - start_time) * grace_factor) * 1000.0  # milliseconds
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

    def exit(self) -> None:
        """Signal the remote workers to exit and terminate the connections."""
        for worker_id in self._workers:
            self._task_endpoint.send_multipart([worker_id, b"EXIT"])
        self._task_endpoint.close()
        self._context.term()


class BatchEnvSampler(EnvSamplerInterface):
    """Facility that samples from multiple copies of an environment in parallel.

    No environment is created here. Instead, it uses a ParallelTaskController to send roll-out requests to a set of
    remote workers and collect results from them.

    Args:
        sampling_parallelism (int, default=1): Parallelism for sampling from the environment.
        port (int, default=DEFAULT_ROLLOUT_PRODUCER_PORT): Network port that the internal ``ParallelTaskController``
            uses to talk to the remote workers.
        min_env_samples (int, default=None): The minimum number of results to collect in one round of remote sampling.
            If it is None, it defaults to the value of ``sampling_parallelism``.
        grace_factor (float, default=None): Factor that determines the additional wait time after receiving the minimum
            required env samples (as determined by ``min_env_samples``). For example, if the minimum required samples
            are received in T seconds, it will allow an additional T * grace_factor seconds to collect the remaining
            results.
        eval_parallelism (int, default=1): Parallelism for policy evaluation on remote workers.
        logger (LoggerV2, default=None): Optional logger for logging key events.
    """

    def __init__(
        self,
        sampling_parallelism: int = 1,
        port: int = DEFAULT_ROLLOUT_PRODUCER_PORT,
        min_env_samples: int = None,
        grace_factor: float = None,
        eval_parallelism: int = 1,
        logger: LoggerV2 = None,
    ) -> None:
        super(BatchEnvSampler, self).__init__()
        self._logger: Union[LoggerV2, DummyLogger] = logger or DummyLogger()
        self._controller = ParallelTaskController(port=port, logger=logger)

        self._sampling_parallelism = sampling_parallelism
        self._min_env_samples = min_env_samples or self._sampling_parallelism
        self._grace_factor = grace_factor
        self._eval_parallelism = eval_parallelism

        self._ep = 0

    def sample(
        self,
        policy_state: Optional[Dict[str, Dict[str, Any]]] = None,
        num_steps: Optional[int] = None,
    ) -> dict:
        """Collect experiences from a set of remote roll-out workers.

        Args:
            policy_state (Dict[str, Any]): Policy state dict. If it is not None, then we need to update all
                policies according to the latest policy states, then start the experience collection.
            num_steps (Optional[int], default=None): Number of environment steps to collect experiences for. If
                it is None, interactions with the (remote) environments will continue until the terminal state is
                reached.

        Returns:
            A dict that contains the collected experiences and additional information.
        """
        # increment episode depending on whether the last episode has concluded
        self._ep += 1
        self._logger.info(f"Collecting roll-out data for episode {self._ep}")
        req = {
            "type": "sample",
            "policy_state": policy_state,
            "index": self._ep,
        }
        num_steps = (
            [None] * self._sampling_parallelism if num_steps is None else _split(num_steps, self._sampling_parallelism)
        )
        results = self._controller.collect(
            req,
            self._sampling_parallelism,
            min_replies=self._min_env_samples,
            grace_factor=self._grace_factor,
            unique_params={"num_steps": num_steps},
        )
        merged_experiences: List[List[ExpElement]] = list(chain(*[res["experiences"] for res in results]))
        return {
            "experiences": merged_experiences,
            "info": [res["info"][0] for res in results],
        }

    def eval(self, policy_state: Dict[str, Dict[str, Any]] = None, num_episodes: int = 1) -> dict:
        req = {
            "type": "eval",
            "policy_state": policy_state,
            "index": self._ep,
        }  # -1 signals test
        results = self._controller.collect(
            req,
            self._eval_parallelism,
            unique_params={"num_eval_episodes": _split(num_episodes, self._eval_parallelism)},
        )
        return {
            "info": [res["info"][0] for res in results],
        }

    def load_policy_state(self, path: str) -> List[str]:
        file_list = os.listdir(path)
        policy_state_dict = {}
        loaded = []
        for file_name in file_list:
            if "non_policy" in file_name or not file_name.endswith(f"_policy.{FILE_SUFFIX}"):  # TODO: remove hardcode
                continue
            policy_name, policy_state = torch.load(os.path.join(path, file_name))
            policy_state_dict[policy_name] = policy_state
            loaded.append(policy_name)

        req = {
            "type": "set_policy_state",
            "policy_state": policy_state_dict,
            "index": self._ep,
        }
        self._controller.collect(req, self._sampling_parallelism)
        return loaded

    def exit(self) -> None:
        self._controller.exit()

    def post_collect(self, ep: int) -> None:
        req = {"type": "post_collect", "index": ep}
        self._controller.collect(req, self._sampling_parallelism)

    def post_evaluate(self, ep: int) -> None:
        req = {"type": "post_evaluate", "index": ep}
        self._controller.collect(req, self._eval_parallelism)

    def monitor_metrics(self) -> float:
        req = {"type": "monitor_metrics", "index": self._ep}
        return float(np.mean(self._controller.collect(req, self._sampling_parallelism)))

    def get_metrics(self) -> dict:
        req = {"type": "get_metrics", "index": self._ep}
        metrics_list = self._controller.collect(req, self._sampling_parallelism)
        req = {"type": "merge_metrics", "metrics_list": metrics_list, "index": self._ep}
        metrics = self._controller.collect(req, 1)[0]
        return metrics
