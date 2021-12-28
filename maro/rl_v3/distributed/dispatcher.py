from typing import Callable, Dict

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.rl_v3.distributed.utils import bytes_to_pyobj, string_to_bytes


class Dispatcher(object):
    def __init__(
        self,
        host: str,
        num_workers: int,
        frontend_port: int = 10000,
        backend_port: int = 10001,
        hash_fn: Callable[[str], int] = hash
    ) -> None:
        # ZMQ sockets and streams
        self._context = Context.instance()
        self._req_socket = self._context.socket(zmq.ROUTER)
        self._req_socket.bind(f"tcp://{host}:{frontend_port}")
        self._req_receiver = ZMQStream(self._req_socket)
        self._route_socket = self._context.socket(zmq.ROUTER)
        self._route_socket.bind(f"tcp://{host}:{backend_port}")
        self._router = ZMQStream(self._route_socket)

        self._event_loop = IOLoop.current()

        # register handlers
        self._req_receiver.on_recv(self._route_request_to_compute_node)
        self._req_receiver.on_send(self.log_send_result)
        self._router.on_recv(self._send_result_to_requester)
        self._router.on_send(self.log_route_request)

        # bookkeeping
        self._num_workers = num_workers
        self._hash_fn = hash_fn
        self._ops2node: Dict[str, int] = {}

    def _route_request_to_compute_node(self, msg: list) -> None:
        ops_name, _, req = msg
        print(f"Received request from {ops_name}")
        if ops_name not in self._ops2node:
            self._ops2node[ops_name] = self._hash_fn(ops_name) % self._num_workers
            print(f"Placing {ops_name} at worker node {self._ops2node[ops_name]}")
        worker_id = f'worker.{self._ops2node[ops_name]}'
        self._router.send_multipart([string_to_bytes(worker_id), b"", ops_name, b"", req])

    def _send_result_to_requester(self, msg: list) -> None:
        worker_id, _, result = msg[:3]
        if result != b"READY":
            self._req_receiver.send_multipart(msg[2:])
        else:
            print(f"{worker_id} ready")

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()

    @staticmethod
    def log_route_request(msg: list, status: object) -> None:
        worker_id, _, ops_name, _, req = msg
        req = bytes_to_pyobj(req)
        print(f"Routed {ops_name}'s request {req['func']} to worker node {worker_id}")

    @staticmethod
    def log_send_result(msg: list, status: object) -> None:
        print(f"Returned result for {msg[0]}")
