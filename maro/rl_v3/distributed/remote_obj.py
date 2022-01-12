# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import socket
from typing import Callable, Tuple

import zmq
# from zmq import Context
from zmq.asyncio import Context

from .utils import bytes_to_pyobj, pyobj_to_bytes 

from maro.utils import Logger

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

    async def send(self, req: dict):
        return await self._socket.send(pyobj_to_bytes(req))

    async def recv(self):
        return bytes_to_pyobj(await self._socket.recv())

    # def send(self, req):
    #     return self._socket.send(pyobj_to_bytes(req))

    # def recv(self):
    #     return bytes_to_pyobj(self._socket.recv())

    def close(self):
        self._socket.close()

    def reset(self):
        self.close()
        self._socket = Context.instance().socket(zmq.DEALER)
        self._socket.setsockopt_string(zmq.IDENTITY, self._name)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._dispatcher_address)
        logger.info(f"reconnected to {self._dispatcher_address}")
        self._retries += 1


def remote_method(obj_name: str, obj_type: str, func_name: str, client) -> Callable:
    async def remote_call(*args, **kwargs) -> object:
        req = {"type": obj_type, "func": func_name, "args": args, "kwargs": kwargs}
        while True:
            try:
                await client.send(req)
                logger.info(f"sent request {func_name} for {obj_name}")
                result = await asyncio.wait_for(client.recv(), timeout=1)
                logger.info(f"received result for request {func_name} for {obj_name}")
                return result
            except asyncio.TimeoutError:
                client.reset()
                logger.info(f"Reset client to retry...")

    return remote_call


class RemoteObj(object):
    def __init__(self, name: str, obj_type: str, dispatcher_address: Tuple[str, int]) -> None:
        assert obj_type in {"rollout", "train"}
        self._name = name
        self._obj_type = obj_type
        self._dispatcher_address = dispatcher_address
        self._client = Client(self._name, dispatcher_address)

    def __getattribute__(self, attr_name: str) -> object:
        # Ignore methods that belong to the parent class
        try:
            return super().__getattribute__(attr_name)
        except AttributeError:
            pass

        # return remote_method(self._name, attr_name, self._dispatcher_address)
        return remote_method(self._name, self._obj_type, attr_name, self._client)
