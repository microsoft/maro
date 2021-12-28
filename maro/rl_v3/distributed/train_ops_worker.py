from typing import Callable, Dict

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from .utils import bytes_to_pyobj, bytes_to_string, pyobj_to_bytes, string_to_bytes


class TrainOpsWorker(object):
    def __init__(
        self,
        idx: int,
        ops_creator: Dict[str, Callable[[str], object]],  # TODO: Callable type?
        router_host: str,
        router_port: int = 10001
    ) -> None:
        # ZMQ sockets and streams
        self._id = f"worker.{idx}"
        self._ops_creator = ops_creator
        self._context = Context.instance()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.identity = string_to_bytes(self._id)
        self._router_address = f"tcp://{router_host}:{router_port}"
        self._socket.connect(self._router_address)
        print(f"Successfully connected to dispatcher at {self._router_address}")
        self._socket.send_multipart([b"", b"READY"])
        self._task_receiver = ZMQStream(self._socket)
        self._event_loop = IOLoop.current()

        # register handlers
        self._task_receiver.on_recv(self._compute)
        self._task_receiver.on_send(self.log_send_result)

        self._ops_dict: Dict[str, object] = {}  # TODO: value type?

    def _compute(self, msg: list) -> None:
        ops_name = bytes_to_string(msg[1])
        req = bytes_to_pyobj(msg[-1])
        assert isinstance(req, dict)

        if ops_name not in self._ops_dict:
            self._ops_dict[ops_name] = self._ops_creator[ops_name](ops_name)
            print(f"Created ops instance {ops_name} at worker {self._id}")

        func_name, args, kwargs = req["func"], req["args"], req["kwargs"]
        func = getattr(self._ops_dict[ops_name], func_name)
        result = func(*args, **kwargs)
        self._task_receiver.send_multipart([b"", msg[1], b"", pyobj_to_bytes(result)])

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()

    @staticmethod
    def log_send_result(msg: list, status: object) -> None:
        print(f"Returning result for {msg[1]}")
