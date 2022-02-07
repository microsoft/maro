# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, deque

from maro.rl.distributed import AbsProxy
from maro.rl.utils.common import bytes_to_pyobj, pyobj_to_bytes
from maro.rl.utils.torch_utils import average_grads
from maro.utils import Logger


class TrainingProxy(AbsProxy):
    def __init__(self, frontend_port: int = 10000, backend_port: int = 10001) -> None:
        super(TrainingProxy, self).__init__(frontend_port=frontend_port, backend_port=backend_port)
        self._available_workers = deque()
        self._worker_ready = False
        self._connected_ops = set()
        self._result_cache = defaultdict(list)
        self._expected_num_results = {}
        self._logger = Logger("TRAIN-PROXY")

    def _route_request_to_compute_node(self, msg: list) -> None:
        if msg[-1] == b"EXIT":
            self._connected_ops.remove(msg[0])
            # if all clients (ops) have signaled exit, tell the workers to terminate
            if not self._connected_ops:
                for worker_id in self._available_workers:
                    self._dispatch_endpoint.send_multipart([worker_id, b"EXIT"])
            return

        self._connected_ops.add(msg[0])
        req = bytes_to_pyobj(msg[-1])
        desired_parallelism = req["desired_parallelism"]
        req["args"] = list(req["args"])
        batch = req["args"][0]
        workers = []
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
