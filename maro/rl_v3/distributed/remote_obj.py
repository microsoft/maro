# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket
from typing import Callable, Tuple

import zmq
from zmq.asyncio import Context

from maro.utils import Logger

from .utils import bytes_to_pyobj, pyobj_to_bytes

logger = Logger("client")


class Client(object):
    def __init__(self, name, dispatcher_address: Tuple[str, int]):
        self._name = name
        self._socket = Context.instance().socket(zmq.DEALER)
        self._socket.setsockopt_string(zmq.IDENTITY, name)
        self._socket.setsockopt(zmq.LINGER, 0)
        host, port = dispatcher_address
        self._dispatcher_ip = socket.gethostbyname(host)
        logger.info(f"dispatcher ip: {self._dispatcher_ip}")
        self._dispatcher_address = f"tcp://{self._dispatcher_ip}:{port}"
        self._socket.connect(self._dispatcher_address)
        logger.info(f"connected to {self._dispatcher_address}")
        self._retries = 0

    async def get_response(self, req: dict):
        await self._socket.send(pyobj_to_bytes(req))
        logger.info(f"sent request {req['func']} for {self._name}")
        while True:
            try:
                result = await self._socket.recv_multipart(flags=zmq.NOBLOCK)
                logger.info(f"received result for request {req['func']} for {self._name}")
                return bytes_to_pyobj(result[0])
            except zmq.ZMQError:
                continue

    def close(self):
        self._socket.close()


def remote_method(obj_type: str, func_name: str, client: Client) -> Callable:
    async def remote_call(*args, **kwargs) -> object:
        req = {"type": obj_type, "func": func_name, "args": args, "kwargs": kwargs}
        return await client.get_response(req)

    return remote_call


class RemoteObj(object):
    def __init__(self, name: str, obj_type: str, dispatcher_address: Tuple[str, int]) -> None:
        assert obj_type in {"rollout", "train"}
        self._name = name
        self._obj_type = obj_type
        self._client = Client(self._name, dispatcher_address)

    def __getattribute__(self, attr_name: str) -> object:
        # Ignore methods that belong to the parent class
        try:
            return super().__getattribute__(attr_name)
        except AttributeError:
            pass

        return remote_method(self._obj_type, attr_name, self._client)

    def exit(self):
        self._client.close()
