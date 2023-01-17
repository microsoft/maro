# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.rl.utils.common import get_own_ip_address


class AbsProxy(object):
    """Abstract proxy class that serves as an intermediary between task producers and task consumers.

    The proxy receives compute tasks from multiple clients, forwards them to a set of back-end workers for
    processing and returns the results to the clients.

    Args:
        frontend_port (int): Network port for communicating with clients (task producers).
        backend_port (int): Network port for communicating with back-end workers (task consumers).
    """

    def __init__(self, frontend_port: int, backend_port: int) -> None:
        super(AbsProxy, self).__init__()

        # ZMQ sockets and streams
        self._context = Context.instance()
        self._req_socket = self._context.socket(zmq.ROUTER)
        self._ip_address = get_own_ip_address()
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
        """Dispatch the task to one or more workers for processing.

        The dispatching strategy should be implemented here.

        Args:
            msg (list): Multi-part message containing task specifications and parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def _send_result_to_requester(self, msg: list) -> None:
        """Return a task result to the client that requested it.

        The result aggregation logic, if applicable, should be implemented here.

        Args:
            msg (list): Multi-part message containing a task result.
        """
        raise NotImplementedError

    def start(self) -> None:
        """Start a Tornado event loop.

        Calling this enters the proxy into an event loop where it starts doing its job.
        """
        self._event_loop.start()

    def stop(self) -> None:
        """Stop the currently running event loop."""
        self._event_loop.stop()
