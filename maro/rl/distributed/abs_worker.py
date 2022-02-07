# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.rl.utils.common import get_ip_address_by_hostname, string_to_bytes
from maro.utils import DummyLogger, Logger


class AbsWorker(object):
    def __init__(
        self,
        idx: int,
        producer_host: str,
        producer_port: int,
        logger: Logger = None
    ) -> None:
        super(AbsWorker, self).__init__()

        self._id = f"worker.{idx}"
        self._logger = logger if logger else DummyLogger()

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
        raise NotImplementedError

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()
