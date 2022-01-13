# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket
from typing import Callable, Dict, Union

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.utils import Logger

from .utils import bytes_to_pyobj, bytes_to_string, pyobj_to_bytes, string_to_bytes


class Worker(object):
    def __init__(
        self,
        type: str,
        idx: int,
        obj_creator: Union[Callable, Dict[str, Callable]],
        router_host: str,
        router_port: int = 10001
    ) -> None:
        self._id = f"{type}_worker.{idx}"
        self._logger = Logger(self._id)
        self._obj_creator = obj_creator

        # ZMQ sockets and streams
        self._context = Context.instance()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.identity = string_to_bytes(self._id)
        self._router_ip = socket.gethostbyname(router_host)
        self._router_address = f"tcp://{self._router_ip}:{router_port}"
        self._socket.connect(self._router_address)
        self._logger.info(f"Connected to dispatcher at {self._router_address}")
        self._task_receiver = ZMQStream(self._socket)
        self._task_receiver.send(b"READY")
        self._event_loop = IOLoop.current()

        # register handlers
        self._task_receiver.on_recv(self._compute)

        self._obj_dict: Dict[str, object] = {}

    def _compute(self, msg: list) -> None:
        obj_name, req = bytes_to_string(msg[0]), bytes_to_pyobj(msg[-1])
        assert isinstance(req, dict)

        if obj_name not in self._obj_dict:
            creator_fn = self._obj_creator if isinstance(self._obj_creator, Callable) else self._obj_creator[obj_name]
            self._obj_dict[obj_name] = creator_fn()
            self._logger.info(f"Created object {obj_name} at worker {self._id}")

        func_name, args, kwargs = req["func"], req["args"], req["kwargs"]
        func = getattr(self._obj_dict[obj_name], func_name)
        result = func(*args, **kwargs)
        self._task_receiver.send_multipart([msg[0], pyobj_to_bytes(result)])

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()
