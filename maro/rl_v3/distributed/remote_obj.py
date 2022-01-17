# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import socket
from typing import Callable, Tuple

import zmq
from zmq.asyncio import Context

from maro.utils import Logger

from .utils import bytes_to_pyobj, pyobj_to_bytes

logger = Logger("client")


class Client(object):
    def __init__(self, name: str, address: Tuple[str, int]) -> None:
        self._name = name
        self._socket = Context.instance().socket(zmq.DEALER)
        self._socket.setsockopt_string(zmq.IDENTITY, name)
        self._socket.setsockopt(zmq.LINGER, 0)
        host, port = address
        self._dispatcher_ip = socket.gethostbyname(host)
        logger.info(f"dispatcher ip: {self._dispatcher_ip}")
        self._address = f"tcp://{self._dispatcher_ip}:{port}"
        self._socket.connect(self._address)
        logger.info(f"connected to {self._address}")
        self._retries = 0

    async def get_response(self, req: dict) -> object:
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


# annotations to indicate that an function / method can be called remotely
def remote():
    def remote_anotate(func):
        return func
    return remote_anotate


def remote_method(obj_state, func_name: str, client: Client) -> Callable:
    async def remote_call(*args, **kwargs) -> object:
        req = {"state": obj_state, "func": func_name, "args": args, "kwargs": kwargs}
        return await client.get_response(req)

    return remote_call


class RemoteObj(object):
    def __init__(self, obj: object, name: str, address: Tuple[str, int]) -> None:
        self._obj = obj
        self._name = name
        self._client = Client(self._name, address)

    def __getattribute__(self, attr_name: str) -> object:
        # Ignore methods that belong to the parent class
        try:
            return super().__getattribute__(attr_name)
        except AttributeError:
            pass

        attr = self._obj.getattr(attr_name)
        if inspect.ismethod(attr) and attr.__name__ != "remote_anotate" and attr.__module__ == "remote_obj":
            return remote_method(self._obj.get_state(), attr_name, self._client)

        return attr

    def exit(self):
        self._client.close()
