# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket
from typing import Callable, Tuple

import zmq
from zmq.asyncio import Context

from .utils import bytes_to_pyobj, pyobj_to_bytes 


class Client(object):
    def __init__(self, name, dispatcher_address: Tuple[str, int]):
        self._client = Context.instance().socket(zmq.REQ)
        self._client.setsockopt_string(zmq.IDENTITY, name)
        self._client.setsockopt(zmq.LINGER, 0)
        host, port = dispatcher_address
        self._dispatcher_ip = socket.gethostbyname(host)
        self._dispatcher_address = f"tcp://{self._dispatcher_ip}:{port}"
        self._client.connect(self._dispatcher_address)

    async def send(self, req: dict):
        return await self._client.send(pyobj_to_bytes(req))

    async def recv(self):
        return bytes_to_pyobj(await self._client.recv())

    def close(self):
        self._client.close()


def remote_method(obj_name: str, obj_type: str, func_name: str, dispatcher_address: str) -> Callable:
    async def remote_call(*args, **kwargs) -> object:
        client = Client(obj_name, dispatcher_address)
        req = {"type": obj_type, "func": func_name, "args": args, "kwargs": kwargs}
        await client.send(req)
        print(f"sent request {func_name} for {obj_name}")
        result = await client.recv()
        client.close()
        print(f"received result for request {func_name} for {obj_name}")
        return result

    return remote_call


class RemoteObj(object):
    def __init__(self, name: str, type_: str, dispatcher_address: Tuple[str, int]) -> None:
        assert type_ in {"rollout", "train"}
        self._name = name
        self._type = type_
        self._dispatcher_address = dispatcher_address

    def __getattribute__(self, attr_name: str) -> object:
        # Ignore methods that belong to the parent class
        try:
            return super().__getattribute__(attr_name)
        except AttributeError:
            pass

        return remote_method(self._name, self._type, attr_name, self._dispatcher_address)
