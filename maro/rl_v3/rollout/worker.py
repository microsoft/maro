# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket
from typing import Callable

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.rl_v3.utils.common import bytes_to_pyobj, pyobj_to_bytes, string_to_bytes
from maro.utils import Logger

from .env_sampler import AbsEnvSampler


class RolloutWorker(object):
    def __init__(
        self,
        idx: int,
        env_sampler_creator: Callable[[], AbsEnvSampler],
        router_host: str,
        router_port: int = 10001
    ) -> None:
        self._id = f"worker.{idx}"
        self._logger = Logger(self._id)
        self._env_sampler_creator = env_sampler_creator

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

        self._env_sampler = env_sampler_creator()

    def _compute(self, msg: list) -> None:
        req = bytes_to_pyobj(msg[-1])
        assert isinstance(req, dict)

        func = getattr(self._env_sampler, req["func"])
        result = func(*req["args"], **req["kwargs"])
        self._receiver.send_multipart([msg[0], pyobj_to_bytes(result)])

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()
