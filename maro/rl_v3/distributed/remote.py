from typing import Tuple

import zmq
from zmq.asyncio import Context

from .utils import bytes_to_pyobj, pyobj_to_bytes, string_to_bytes


def remote_method(ops_name, func_name: str, dispatcher_address):
    async def remote_call(*args, **kwargs):
        req = {"func": func_name, "args": args, "kwargs": kwargs}
        context = Context.instance()
        sock = context.socket(zmq.REQ)
        sock.identity = string_to_bytes(ops_name)
        sock.connect(dispatcher_address)
        sock.send(pyobj_to_bytes(req))
        print(f"sent request {func_name} for {ops_name}")
        result = bytes_to_pyobj(await sock.recv())
        print(f"result for request {func_name} for {ops_name}: {result}")
        sock.close()
        return result

    return remote_call


class RemoteOps(object):
    def __init__(self, ops_name, dispatcher_address: Tuple[str, int]):
        self._ops_name = ops_name
        # self._functions = {name for name, _ in inspect.getmembers(train_op_cls, lambda attr: inspect.isfunction(attr))}
        host, port = dispatcher_address
        self._dispatcher_address = f"tcp://{host}:{port}"

    def __getattribute__(self, attr_name: str):
        # Ignore methods that belong to the parent class
        try:
            return super().__getattribute__(attr_name)
        except AttributeError:
            pass

        return remote_method(self._ops_name, attr_name, self._dispatcher_address)
