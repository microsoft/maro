# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import deque

from maro.rl.distributed import AbsDispatcher


class TrainOpsDispatcher(AbsDispatcher):
    """Train ops dispatcher.

    Args:
        frontend_port (int, default=10000): Frontend port, which is used to communicate with requesters.
        backend_port (int, default=10001): Backend port, which is used to communicate with workers.
    """
    def __init__(self, frontend_port: int = 10000, backend_port: int = 10001) -> None:
        super(TrainOpsDispatcher, self).__init__(frontend_port=frontend_port, backend_port=backend_port)
        self._available_workers = deque()
        self._worker_ready = False

    def _route_request_to_compute_node(self, msg: list) -> None:
        worker_id = self._available_workers.popleft()
        self._dispatch_endpoint.send_multipart([worker_id, msg[0], msg[-1]])
        if not self._available_workers:
            # stop receiving compute requests until at least one worker becomes available
            self._workers_ready = False
            self._req_endpoint.stop_on_recv()

    def _send_result_to_requester(self, msg: list) -> None:
        if msg[1] != b"READY":
            self._req_endpoint.send_multipart(msg[1:])

        self._available_workers.append(msg[0])
        self._worker_ready = True
        self._req_endpoint.on_recv(self._route_request_to_compute_node)
