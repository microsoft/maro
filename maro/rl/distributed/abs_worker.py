# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket
from abc import abstractmethod

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.rl.utils.common import string_to_bytes
from maro.utils import DummyLogger, Logger


class AbsWorker(object):
    def __init__(
        self,
        idx: int,
        router_host: str,
        router_port: int = 10001,
        logger: Logger = None
    ) -> None:
        super(AbsWorker, self).__init__()

        self._id = f"worker.{idx}"
        self._logger = DummyLogger() if logger is None else logger

        # ZMQ sockets and streams
        self._context = Context.instance()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.identity = string_to_bytes(self._id)

        self._router_ip = socket.gethostbyname(router_host)
        self._router_address = f"tcp://{self._router_ip}:{router_port}"
        self._socket.connect(self._router_address)
        self._logger.info(f"Connected to dispatcher at {self._router_address}")

        self._receiver = ZMQStream(self._socket)
        self._receiver.send(b"READY")

        self._event_loop = IOLoop.current()

        # register handlers
        self._receiver.on_recv(self._compute)

    @abstractmethod
    def _compute(self, msg: list) -> None:
        raise NotImplementedError

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()
