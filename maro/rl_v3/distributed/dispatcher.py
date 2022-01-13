# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket
from typing import Callable, Dict

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.utils import Logger

from .utils import bytes_to_pyobj, bytes_to_string, string_to_bytes




class Dispatcher(object):
    def __init__(
        self,
        num_workers: int,
        frontend_port: int = 10000,
        backend_port: int = 10001,
        hash_fn: Callable[[str], int] = hash
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

        # bookkeeping
        self._num_workers = num_workers
        self._num_checkedin_workers = 0
        self._hash_fn = hash_fn
        self._obj2node: Dict[str, int] = {}

        self._logger = Logger("dispatcher")

    def _route_request_to_compute_node(self, msg: list) -> None:
        obj_name = bytes_to_string(msg[0])
        req = bytes_to_pyobj(msg[-1])
        obj_type = req["type"]
        if obj_name not in self._obj2node:
            worker_idx = self._hash_fn(obj_name) % self._num_workers
            worker_id = f"{obj_type}_worker.{worker_idx}"
            self._obj2node[obj_name] = worker_id
            self._logger.info(f"Placing {obj_name} at worker node {self._obj2node[obj_name]}")
        else:
            worker_id = self._obj2node[obj_name]

        self._router.send_multipart([string_to_bytes(worker_id), msg[0], msg[-1]])

    def _send_result_to_requester(self, msg: list) -> None:
        worker_id = msg[0]
        if msg[1] == b"READY":
            self._logger.info(f"{bytes_to_string(worker_id)} ready")
            self._num_checkedin_workers += 1
            if self._num_checkedin_workers == self._num_workers:
                self._req_receiver.on_recv(self._route_request_to_compute_node)
        else:
            self._req_receiver.send_multipart(msg[1:])

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()
