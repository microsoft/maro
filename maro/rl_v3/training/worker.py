# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket
from typing import Callable, Dict

import zmq
from tornado.ioloop import IOLoop
from zmq import Context
from zmq.eventloop.zmqstream import ZMQStream

from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.utils.common import bytes_to_pyobj, bytes_to_string, pyobj_to_bytes, string_to_bytes
from maro.utils import Logger

from .train_ops import AbsTrainOps
from .trainer import AbsTrainer


class TrainOpsWorker(object):
    def __init__(
        self,
        idx: int,
        policy_creator: Dict[str, Callable[[str], RLPolicy]],
        trainer_creator: Dict[str, Callable[[str], AbsTrainer]],
        router_host: str,
        router_port: int = 10001
    ) -> None:
        self._id = f"worker.{idx}"
        self._logger = Logger(self._id)

        self._policy_creator = policy_creator
        self._trainer_creator = trainer_creator
        self._trainer_dict: Dict[str, AbsTrainer] = {}

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

        self._ops_dict: Dict[str, AbsTrainOps] = {}

    def _compute(self, msg: list) -> None:
        ops_name, req = bytes_to_string(msg[0]), bytes_to_pyobj(msg[-1])
        assert isinstance(req, dict)

        if ops_name not in self._ops_dict:
            trainer_name = ops_name.split(".")[0]
            if trainer_name not in self._trainer_dict:
                trainer = self._trainer_creator[trainer_name](trainer_name)
                trainer.register_policy_creator(self._policy_creator)
                self._trainer_dict[trainer_name] = trainer

            self._ops_dict[ops_name] = self._trainer_dict[trainer_name].get_local_ops_by_name(ops_name)
            self._logger.info(f"Created ops {ops_name} at {self._id}")

        self._ops_dict[ops_name].set_state(req["state"])
        func = getattr(self._ops_dict[ops_name], req["func"])
        result = func(*req["args"], **req["kwargs"])
        self._task_receiver.send_multipart([msg[0], pyobj_to_bytes(result)])

    def start(self) -> None:
        self._event_loop.start()

    def stop(self) -> None:
        self._event_loop.stop()
