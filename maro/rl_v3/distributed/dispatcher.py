# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket
from collections import deque

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from .utils import string_to_bytes


class Dispatcher():
    def __init__(
        self,
        frontend_port: int = 10000,
        backend_port: int = 10001
    ) -> None:
        # ZMQ sockets and streams
        self._context = Context.instance()
        self._req_socket = self._context.socket(zmq.ROUTER)
        self._ip_address = socket.gethostbyname(socket.gethostname())
        self._req_socket.bind(f"tcp://{self._ip_address}:{frontend_port}")
        self._req_receiver = ZMQStream(self._req_socket)
        self._route_socket = self._context.socket(zmq.ROUTER)
        self._route_socket.bind(f"tcp://{self._ip_address}:{backend_port}")
        self._router = ZMQStream(self._route_socket)
        self._event_loop = IOLoop.current()

        # register handlers
        self._router.on_recv(self._send_result_to_requester)

        # workers
        self._available_workers = deque()
        self._worker_ready = False

    def _route_request_to_compute_node(self, msg: list) -> None:
        worker_id = self._available_workers.popleft()
        self._router.send_multipart([string_to_bytes(worker_id), msg[0], msg[-1]])
        if not self._available_workers:
            # stop receiving compute requests until at least one worker becomes available
            self._workers_ready = False
            self._req_receiver.stop_on_recv()

    def _send_result_to_requester(self, msg: list) -> None:
        if msg[1] != b"READY":
            self._req_receiver.send_multipart(msg[1:])

        self._available_workers.append(msg[0])
        if not self._worker_ready:
            self._worker_ready = True
            self._req_receiver.on_recv(self._route_request_to_compute_node)

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()
