# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.rl_v3.utils.common import get_ip_address


class AbsDispatcher(object):
    def __init__(self, frontend_port: int, backend_port) -> None:
        super(AbsDispatcher, self).__init__()

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

    @abstractmethod
    def _route_request_to_compute_node(self, msg: list) -> None:
        raise NotImplementedError

    @abstractmethod
    def _send_result_to_requester(self, msg: list) -> None:
        raise NotImplementedError

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()
