# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Dict

from maro.rl.distributed import AbsWorker
from maro.rl.policy import RLPolicy
from maro.rl.utils.common import bytes_to_pyobj, bytes_to_string, pyobj_to_bytes

from .train_ops import AbsTrainOps
from .trainer import AbsTrainer


class TrainOpsWorker(AbsWorker):
    def __init__(
        self,
        idx: int,
        policy_creator: Dict[str, Callable[[str], RLPolicy]],
        trainer_creator: Dict[str, Callable[[str], AbsTrainer]],
        router_host: str,
        router_port: int = 10001
    ) -> None:
        super(TrainOpsWorker, self).__init__(
            idx=idx, router_host=router_host, router_port=router_port
        )

        self._policy_creator = policy_creator
        self._trainer_creator = trainer_creator
        self._trainer_dict: Dict[str, AbsTrainer] = {}

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
        self._receiver.send_multipart([msg[0], pyobj_to_bytes(result)])
