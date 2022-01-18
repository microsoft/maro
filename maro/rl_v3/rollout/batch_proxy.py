# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.rl_v3.utils.common import bytes_to_pyobj, get_ip_address, pyobj_to_bytes


class BatchProxy(object):
    def __init__(self, num_workers: int, frontend_port: int = 20000, backend_port: int = 20001) -> None:
        self._num_workers = num_workers

        # ZMQ sockets and streams
        self._context = Context.instance()
        self._req_socket = self._context.socket(zmq.ROUTER)
        self._ip_address = get_ip_address()
        self._req_socket.bind(f"tcp://{self._ip_address}:{frontend_port}")
        self._req_endpoint = ZMQStream(self._req_socket)
        self._dispatch_socket = self._context.socket(zmq.ROUTER)
        self._dispatch_socket.bind(f"tcp://{self._ip_address}:{backend_port}")
        self._dispatcher = ZMQStream(self._dispatch_socket)
        self._event_loop = IOLoop.current()

        # register handlers
        self._dispatcher.on_recv(self._send_result_to_requester)

        # workers
        self._workers = []

    def _route_request_to_compute_node(self, msg: list) -> None:
        batch_req = bytes_to_pyobj(msg[-1])
        parallelism = batch_req["parallelism"]
        req = {"func": batch_req["type"], "args": (), "kwargs": {"policy_state": batch_req["policy_state"]}}
        if "num_steps" in batch_req:
            req["kwargs"]["num_steps"] = batch_req["num_steps"]
        for worker_id in self._workers[:parallelism]:
            self._dispatcher.send_multipart([worker_id, msg[0], pyobj_to_bytes(req)])

    def _send_result_to_requester(self, msg: list) -> None:
        if msg[1] != b"READY":
            self._req_endpoint.send_multipart(msg[1:])
        else:
            self._workers.append(msg[0])
            if len(self._workers) == self._num_workers:
                self._req_endpoint.on_recv(self._route_request_to_compute_node)

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()
