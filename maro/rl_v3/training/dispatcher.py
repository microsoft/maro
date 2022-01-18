# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import deque

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.rl_v3.utils.common import get_ip_address


class Dispatcher(object):
    def __init__(self, frontend_port: int = 10000, backend_port: int = 10001) -> None:
        # ZMQ sockets and streams
        self._context = Context.instance()
        self._req_socket = self._context.socket(zmq.ROUTER)
        self._ip_address = get_ip_address()
        self._req_socket.bind(f"tcp://{self._ip_address}:{frontend_port}")
        self._req_endpoint = ZMQStream(self._req_socket)
        self._dispatch_socket = self._context.socket(zmq.ROUTER)
        self._dispatch_socket.bind(f"tcp://{self._ip_address}:{backend_port}")
        self._dispatch_endpoint = ZMQStream(self._dispatch_socket)
        self._event_loop = IOLoop.current()

        # register handlers
        self._dispatch_endpoint.on_recv(self._send_result_to_requester)

        # workers
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
        if not self._worker_ready:
            self._worker_ready = True
            self._req_endpoint.on_recv(self._route_request_to_compute_node)

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()
