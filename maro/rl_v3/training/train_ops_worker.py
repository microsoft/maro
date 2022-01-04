# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Dict

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.utils.distributed import bytes_to_pyobj, bytes_to_string, pyobj_to_bytes, string_to_bytes

from .abs_train_ops import AbsTrainOps
from .trainer import AbsTrainer


class TrainOpsWorker(object):
    def __init__(
        self,
        idx: int,
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        trainer_param_dict: Dict[str, tuple],
        router_host: str,
        router_port: int = 10001
    ) -> None:
        # ZMQ sockets and streams
        self._id = f"worker.{idx}"
        self._get_policy_func_dict = get_policy_func_dict
        self._trainer_param_dict = trainer_param_dict
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

        self._trainer_dict: Dict[str, AbsTrainer] = {}
        self._ops_dict: Dict[str, AbsTrainOps] = {}  # TODO: value type?

    def _compute(self, msg: list) -> None:
        ops_name = bytes_to_string(msg[1])
        req = bytes_to_pyobj(msg[-1])
        assert isinstance(req, dict)

        if ops_name not in self._ops_dict:
            self._ops_dict[ops_name] = self._create_local_ops(ops_name)
            print(f"Created ops instance {ops_name} at worker {self._id}")

        func_name, args, kwargs = req["func"], req["args"], req["kwargs"]
        func = getattr(self._ops_dict[ops_name], func_name)
        result = func(*args, **kwargs)
        self._task_receiver.send_multipart([b"", msg[1], b"", pyobj_to_bytes(result)])

    def _create_local_ops(self, name: str):
        trainer_name = name.split(".")[0]
        if trainer_name not in self._trainer_dict:
            trainer_cls, param = self._trainer_param_dict[trainer_name]
            get_policy_func_dict = {
                name: func for name, func in self._get_policy_func_dict.items() if name.startswith(trainer_name)
            }
            self._trainer_dict[trainer_name] = trainer_cls(
                name=trainer_name, get_policy_func_dict=get_policy_func_dict, **param
            )
        return self._trainer_dict[trainer_name].create_local_ops(name=name)

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()

    @staticmethod
    def log_send_result(msg: list, status: object) -> None:
        print(f"Returning result for {msg[1]}")
