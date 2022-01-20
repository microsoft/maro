# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl_v3.distributed import AbsDispatcher
from maro.rl_v3.utils.common import bytes_to_pyobj, pyobj_to_bytes


class RolloutDispatcher(AbsDispatcher):
    def __init__(self, num_workers: int, frontend_port: int = 20000, backend_port: int = 20001) -> None:
        super(RolloutDispatcher, self).__init__(frontend_port=frontend_port, backend_port=backend_port)
        self._num_workers = num_workers
        self._workers = []

    def _route_request_to_compute_node(self, msg: list) -> None:
        batch_req = bytes_to_pyobj(msg[-1])
        assert isinstance(batch_req, dict)

        parallelism = batch_req["parallelism"]
        req = {"func": batch_req["type"], "args": (), "kwargs": {"policy_state": batch_req["policy_state"]}}
        if "num_steps" in batch_req:
            req["kwargs"]["num_steps"] = batch_req["num_steps"]
        for worker_id in self._workers[:parallelism]:
            self._dispatch_endpoint.send_multipart([worker_id, msg[0], pyobj_to_bytes(req)])

    def _send_result_to_requester(self, msg: list) -> None:
        if msg[1] != b"READY":
            self._req_endpoint.send_multipart(msg[1:])
        else:
            self._workers.append(msg[0])
            if len(self._workers) == self._num_workers:
                self._req_endpoint.on_recv(self._route_request_to_compute_node)
