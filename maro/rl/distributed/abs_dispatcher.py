# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.rl.utils.common import get_ip_address


class AbsDispatcher(object):
    """Dispatcher. Dispatcher receives job requests, dispatch job requests to proper workers, and then forward the
        results from the worker to the requester.

    Args:
        frontend_port (int): Frontend port, which is used to communicate with requesters.
        backend_port (int): Backend port, which is used to communicate with workers.
    """
    def __init__(self, frontend_port: int, backend_port: int) -> None:
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
        """Dispatch the job request to workers.

        Args:
            msg (list): Message.
        """
        raise NotImplementedError

    @abstractmethod
    def _send_result_to_requester(self, msg: list) -> None:
        """Send the results from workers to requesters.

        Args:
            msg (list): Message.
        """
        raise NotImplementedError

    def start(self) -> None:
        """Start the dispatcher.
        """
        self._event_loop.start()

    def stop(self) -> None:
        """Stop the dispatcher.
        """
        self._event_loop.stop()
