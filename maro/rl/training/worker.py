# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import typing
from typing import Dict

from maro.rl.distributed import DEFAULT_TRAINING_BACKEND_PORT, AbsWorker
from maro.rl.training import SingleAgentTrainer
from maro.rl.utils.common import bytes_to_pyobj, bytes_to_string, pyobj_to_bytes
from maro.utils import LoggerV2

from .train_ops import AbsTrainOps
from .trainer import AbsTrainer, MultiAgentTrainer

if typing.TYPE_CHECKING:
    from maro.rl.rl_component.rl_component_bundle import RLComponentBundle


class TrainOpsWorker(AbsWorker):
    """Worker that executes methods defined in a subclass of ``AbsTrainOps`` and annotated as "remote" on demand.

    Args:
        idx (int): Integer identifier for the worker. It is used to generate an internal ID, "worker.{idx}",
            so that the proxy can keep track of its connection status.
        rl_component_bundle (RLComponentBundle): Resources to launch the RL workflow.
        producer_host (str): IP address of the proxy host to connect to.
        producer_port (int, default=10001): Port of the proxy host to connect to.
    """

    def __init__(
        self,
        idx: int,
        rl_component_bundle: RLComponentBundle,
        producer_host: str,
        producer_port: int = None,
        logger: LoggerV2 = None,
    ) -> None:
        super(TrainOpsWorker, self).__init__(
            idx=idx,
            producer_host=producer_host,
            producer_port=producer_port if producer_port is not None else DEFAULT_TRAINING_BACKEND_PORT,
            logger=logger,
        )

        self._rl_component_bundle = rl_component_bundle
        self._trainer_dict: Dict[str, AbsTrainer] = {}

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

            trainer_dict: Dict[str, AbsTrainer] = {
                trainer.name: trainer for trainer in self._rl_component_bundle.trainers
            }

            if ops_name not in self._ops_dict:
                trainer_name = self._rl_component_bundle.policy_trainer_mapping[ops_name]
                if trainer_name not in self._trainer_dict:
                    trainer = trainer_dict[trainer_name]
                    trainer.register_policies(
                        policies=self._rl_component_bundle.policies,
                        policy_trainer_mapping=self._rl_component_bundle.policy_trainer_mapping,
                    )
                    self._trainer_dict[trainer_name] = trainer

                trainer = self._trainer_dict[trainer_name]
                if isinstance(trainer, SingleAgentTrainer):
                    self._ops_dict[ops_name] = trainer.get_local_ops()
                else:
                    assert isinstance(trainer, MultiAgentTrainer)
                    self._ops_dict[ops_name] = trainer.get_local_ops(ops_name)
                self._logger.info(f"Created ops {ops_name} at {self._id}")

            self._ops_dict[ops_name].set_state(req["state"])
            func = getattr(self._ops_dict[ops_name], req["func"])
            result = func(*req["args"], **req["kwargs"])
            self._stream.send_multipart([msg[0], pyobj_to_bytes(result)])
