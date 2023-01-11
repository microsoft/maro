# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Union

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.rl.utils.common import get_ip_address_by_hostname, string_to_bytes
from maro.utils import DummyLogger, LoggerV2


class AbsWorker(object):
    """Abstract worker class to process a task in distributed fashion.

    Args:
        idx (int): Integer identifier for the worker. It is used to generate an internal ID, "worker.{idx}",
            so that the task producer can keep track of its connection status.
        producer_host (str): IP address of the task producer host to connect to.
        producer_port (int): Port of the task producer host to connect to.
        logger (Logger, default=None): The logger of the workflow.
    """

    def __init__(
        self,
        idx: int,
        producer_host: str,
        producer_port: int,
        logger: LoggerV2 = None,
    ) -> None:
        super(AbsWorker, self).__init__()

        self._id = f"worker.{idx}"
        self._logger: Union[LoggerV2, DummyLogger] = logger if logger else DummyLogger()

        # ZMQ sockets and streams
        self._context = Context.instance()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.identity = string_to_bytes(self._id)

        self._producer_ip = get_ip_address_by_hostname(producer_host)
        self._producer_address = f"tcp://{self._producer_ip}:{producer_port}"
        self._socket.connect(self._producer_address)
        self._logger.info(f"Connected to producer at {self._producer_address}")

        self._stream = ZMQStream(self._socket)
        self._stream.send(b"READY")

        self._event_loop = IOLoop.current()

        # register handlers
        self._stream.on_recv(self._compute)

    @abstractmethod
    def _compute(self, msg: list) -> None:
        """The task processing logic should be implemented here.

        Args:
            msg (list): Multi-part message containing task specifications and parameters.
        """
        raise NotImplementedError

    def start(self) -> None:
        """Start a Tornado event loop.

        Calling this enters the worker into an event loop where it starts doing its job.
        """
        self._event_loop.start()

    def stop(self) -> None:
        """Stop the currently running event loop."""
        self._event_loop.stop()
