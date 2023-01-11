# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, deque
from typing import Deque

from maro.rl.distributed import DEFAULT_TRAINING_BACKEND_PORT, DEFAULT_TRAINING_FRONTEND_PORT, AbsProxy
from maro.rl.utils.common import bytes_to_pyobj, pyobj_to_bytes
from maro.rl.utils.torch_utils import average_grads
from maro.utils import LoggerV2


class TrainingProxy(AbsProxy):
    """Intermediary between trainers and workers.

    The proxy receives compute tasks from multiple ``AbsTrainOps`` instances, forwards them to a set of back-end
    ``TrainOpsWorker``s to be processed and returns the results to the clients.

    Args:
        frontend_port (int, default=10000): Network port for communicating with clients (task producers).
        backend_port (int, default=10001): Network port for communicating with back-end workers (task consumers).
    """

    def __init__(self, frontend_port: int = None, backend_port: int = None) -> None:
        super(TrainingProxy, self).__init__(
            frontend_port=frontend_port if frontend_port is not None else DEFAULT_TRAINING_FRONTEND_PORT,
            backend_port=backend_port if backend_port is not None else DEFAULT_TRAINING_BACKEND_PORT,
        )
        self._available_workers: Deque = deque()
        self._worker_ready: bool = False
        self._connected_ops: set = set()
        self._result_cache: dict = defaultdict(list)
        self._expected_num_results: dict = {}
        self._logger = LoggerV2("TRAIN-PROXY")

    def _route_request_to_compute_node(self, msg: list) -> None:
        """
        Here we use a least-recently-used (LRU) routing strategy to select workers for a task while making the best
        effort to satisfy the task's desired parallelism. For example, consider a task that specifies a desired
        parallelism K (for gradient computation). If there are more than K workers in the ``_available_workers`` queue,
        the first, i.e., the least recently used, K of them will be selected to process the task. If there are fewer
        than K workers in the queue, all workers will be popped from the queue to process the task. In this case, the
        desired parallelism cannot be satisfied, but waiting is avoided.
        """
        if msg[-1] == b"EXIT":
            self._connected_ops.remove(msg[0])
            # if all clients (ops) have signaled exit, tell the workers to terminate
            if not self._connected_ops:
                for worker_id in self._available_workers:
                    self._dispatch_endpoint.send_multipart([worker_id, b"EXIT"])
            return

        self._connected_ops.add(msg[0])
        req = bytes_to_pyobj(msg[-1])
        assert isinstance(req, dict)

        desired_parallelism = req["desired_parallelism"]
        req["args"] = list(req["args"])
        batch = req["args"][0]
        workers: list = []
        while len(workers) < desired_parallelism and self._available_workers:
            workers.append(self._available_workers.popleft())

        self._expected_num_results[msg[0]] = len(workers)
        for worker_id, sub_batch in zip(workers, batch.split(len(workers))):
            req["args"][0] = sub_batch
            self._dispatch_endpoint.send_multipart([worker_id, msg[0], pyobj_to_bytes(req)])

        if not self._available_workers:
            # stop receiving compute requests until at least one worker becomes available
            self._workers_ready = False
            self._req_endpoint.stop_on_recv()

    def _send_result_to_requester(self, msg: list) -> None:
        if msg[1] == b"EXIT_ACK":
            self._logger.info("Exiting event loop...")
            self.stop()
            return

        if msg[1] != b"READY":
            ops_name = msg[1]
            self._result_cache[ops_name].append(bytes_to_pyobj(msg[-1]))
            if len(self._result_cache[ops_name]) == self._expected_num_results[ops_name]:
                aggregated_result = average_grads(self._result_cache[ops_name])
                self._logger.info(f"Aggregated {len(self._result_cache[ops_name])} results for {ops_name}")
                self._result_cache[ops_name].clear()
                self._req_endpoint.send_multipart([ops_name, pyobj_to_bytes(aggregated_result)])

        self._available_workers.append(msg[0])
        self._worker_ready = True
        self._req_endpoint.on_recv(self._route_request_to_compute_node)
