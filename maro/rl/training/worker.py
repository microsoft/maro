# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Dict

from maro.rl.distributed import AbsWorker
from maro.rl.policy import RLPolicy
from maro.rl.utils.common import bytes_to_pyobj, bytes_to_string, pyobj_to_bytes
from maro.utils import Logger

from .train_ops import AbsTrainOps
from maro.rl.training.algorithms.abs_algorithm import AbsAlgorithm


class TrainOpsWorker(AbsWorker):
    """Worker that executes methods defined in a subclass of ``AbsTrainOps`` and annotated as "remote" on demand.

    Args:
        idx (int): Integer identifier for the worker. It is used to generate an internal ID, "worker.{idx}",
            so that the proxy can keep track of its connection status.
        policy_creator (Dict[str, Callable[[str], RLPolicy]]): User-defined function registry that can be used to create
            an "RLPolicy" instance with a name in the registry. This is required to create train ops instances.
        algorithm_instance_creator (Dict[str, Callable[[str], AbsAlgorithm]]): User-defined function registry that
            can be used to create an "AbsAlgorithm" instance with a name in the registry. This is required to
            create train ops instances.
        producer_host (str): IP address of the proxy host to connect to.
        producer_port (int, default=10001): Port of the proxy host to connect to.
    """

    def __init__(
        self,
        idx: int,
        policy_creator: Dict[str, Callable[[str], RLPolicy]],
        algorithm_instance_creator: Dict[str, Callable[[str], AbsAlgorithm]],
        producer_host: str,
        producer_port: int = 10001,
        logger: Logger = None,
    ) -> None:
        super(TrainOpsWorker, self).__init__(
            idx=idx, producer_host=producer_host, producer_port=producer_port, logger=logger,
        )

        self._policy_creator = policy_creator
        self._algo_inst_creator = algorithm_instance_creator
        self._algo_inst_dict: Dict[str, AbsAlgorithm] = {}

        self._ops_dict: Dict[str, AbsTrainOps] = {}

    def _compute(self, msg: list) -> None:
        """Execute a method defined by some train ops and annotated as "remote".

        Args:
            msg (list): Multi-part message containing task specifications and parameters.
        """
        if msg[-1] == b"EXIT":
            self._stream.send(b"EXIT_ACK")
            self.stop()
        else:
            ops_name, req = bytes_to_string(msg[0]), bytes_to_pyobj(msg[-1])
            assert isinstance(req, dict)

            if ops_name not in self._ops_dict:
                algo_inst_name = ops_name.split(".")[0]
                if algo_inst_name not in self._algo_inst_dict:
                    algo_inst = self._algo_inst_creator[algo_inst_name](algo_inst_name)
                    algo_inst.register_policy_creator(self._policy_creator)
                    self._algo_inst_dict[algo_inst_name] = algo_inst

                self._ops_dict[ops_name] = self._algo_inst_dict[algo_inst_name].get_local_ops_by_name(ops_name)
                self._logger.info(f"Created ops {ops_name} at {self._id}")

            self._ops_dict[ops_name].set_state(req["state"])
            func = getattr(self._ops_dict[ops_name], req["func"])
            result = func(*req["args"], **req["kwargs"])
            self._stream.send_multipart([msg[0], pyobj_to_bytes(result)])
