# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Dict

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from .utils import bytes_to_pyobj, bytes_to_string, pyobj_to_bytes, string_to_bytes


class Dispatcher(object):
    def __init__(
        self,
        host: str,
        num_workers: int,
        frontend_port: int = 10000,
        backend_port: int = 10001,
        hash_fn: Callable[[str], int] = hash
    ) -> None:
        # ZMQ sockets and streams
        self._context = Context.instance()
        self._req_socket = self._context.socket(zmq.ROUTER)
        self._req_socket.bind(f"tcp://{host}:{frontend_port}")
        self._req_receiver = ZMQStream(self._req_socket)
        self._route_socket = self._context.socket(zmq.ROUTER)
        self._route_socket.bind(f"tcp://{host}:{backend_port}")
        self._router = ZMQStream(self._route_socket)

        self._event_loop = IOLoop.current()

        # register handlers
        self._req_receiver.on_recv(self._route_request_to_compute_node)
        self._req_receiver.on_send(self.log_send_result)
        self._router.on_recv(self._send_result_to_requester)
        self._router.on_send(self.log_route_request)

        # bookkeeping
        self._num_workers = num_workers
        self._hash_fn = hash_fn
        self._obj2node: Dict[str, int] = {}

    def _route_request_to_compute_node(self, msg: list) -> None:
        obj_name = bytes_to_string(msg[0])
        req = bytes_to_pyobj(msg[-1])
        obj_type = req["type"]
        print(f"Received request {req['func']} from {obj_name}")
        if obj_name not in self._obj2node:
            self._obj2node[obj_name] = self._hash_fn(obj_name) % self._num_workers
            print(f"Placing {obj_name} at worker node {self._obj2node[obj_name]}")
        worker_id = f'{obj_type}_worker.{self._obj2node[obj_name]}'
        self._router.send_multipart(
            [string_to_bytes(worker_id), b"", string_to_bytes(obj_name), b"", pyobj_to_bytes(req)]
        )

    def _send_result_to_requester(self, msg: list) -> None:
        worker_id, _, result = msg[:3]
        if result != b"READY":
            self._req_receiver.send_multipart(msg[2:])
        else:
            print(f"{worker_id} ready")

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()

    @staticmethod
    def log_route_request(msg: list, status: object) -> None:
        worker_id, _, obj_name, _, req = msg
        req = bytes_to_pyobj(req)
        print(f"Routed {obj_name}'s request {req['func']} to worker node {worker_id}")

    @staticmethod
    def log_send_result(msg: list, status: object) -> None:
        print(f"Returned result for {msg[0]}")
