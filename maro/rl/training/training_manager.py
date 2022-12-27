# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import collections
import os
import typing
from typing import Any, Dict, Iterable, List, Tuple

from maro.rl.rollout import ExpElement
from maro.rl.training import SingleAgentTrainer
from maro.utils import LoggerV2
from maro.utils.exception.rl_toolkit_exception import MissingTrainer

from .trainer import AbsTrainer, MultiAgentTrainer

if typing.TYPE_CHECKING:
    from maro.rl.rl_component.rl_component_bundle import RLComponentBundle


class TrainingManager(object):
    """
    Training manager. Manage and schedule all trainers to train policies.

    Args:
        rl_component_bundle (RLComponentBundle): Resources to launch the RL workflow.
        explicit_assign_device (bool, default=False): Whether to assign policy to its device in the training manager.
        proxy_address (Tuple[str, int], default=None): Address of the training proxy. If it is not None,
            it is registered to all trainers, which in turn create `RemoteOps` for distributed training.
        logger (LoggerV2, default=None): A logger for logging key events.
    """

    def __init__(
        self,
        rl_component_bundle: RLComponentBundle,
        explicit_assign_device: bool = False,
        proxy_address: Tuple[str, int] = None,
        logger: LoggerV2 = None,
    ) -> None:
        super(TrainingManager, self).__init__()

        self._proxy_address = proxy_address

        self._trainer_dict: Dict[str, AbsTrainer] = {}
        for trainer in rl_component_bundle.trainers:
            if self._proxy_address:
                trainer.set_proxy_address(self._proxy_address)
            trainer.register_agent2policy(
                agent2policy=rl_component_bundle.trainable_agent2policy,
                policy_trainer_mapping=rl_component_bundle.policy_trainer_mapping,
            )
            trainer.register_policies(
                policies=rl_component_bundle.policies,
                policy_trainer_mapping=rl_component_bundle.policy_trainer_mapping,
            )
            trainer.register_logger(logger)
            trainer.build()  # `build()` must be called after `register_policies()`
            self._trainer_dict[trainer.name] = trainer

        # User-defined allocation of compute devices, i.e., GPU's to the trainer ops
        if explicit_assign_device:
            for policy_name, device_name in rl_component_bundle.device_mapping.items():
                trainer = self._trainer_dict[rl_component_bundle.policy_trainer_mapping[policy_name]]

                if isinstance(trainer, SingleAgentTrainer):
                    ops = trainer.ops
                else:
                    assert isinstance(trainer, MultiAgentTrainer)
                    ops = trainer.ops_dict[policy_name]
                ops.to_device(device_name)

        self._agent2trainer: Dict[Any, str] = {}
        for agent_name, policy_name in rl_component_bundle.trainable_agent2policy.items():
            trainer_name = rl_component_bundle.policy_trainer_mapping[policy_name]
            if trainer_name not in self._trainer_dict:
                raise MissingTrainer(f"trainer {trainer_name} does not exist")
            self._agent2trainer[agent_name] = trainer_name

    def train_step(self) -> None:
        if self._proxy_address:

            async def train_step() -> Iterable:
                return await asyncio.gather(
                    *[trainer_.train_step_as_task() for trainer_ in self._trainer_dict.values()]
                )

            asyncio.run(train_step())
        else:
            for trainer in self._trainer_dict.values():
                trainer.train_step()

    def get_policy_state(self) -> Dict[str, dict]:
        """Get policies' states.

        Returns:
            A double-deck dict with format: {trainer_name: {policy_name: policy_state}}
        """
        policy_states: Dict[str, dict] = {}
        for trainer in self._trainer_dict.values():
            policy_states.update(trainer.get_policy_state())
        return policy_states

    def record_experiences(self, experiences: List[List[ExpElement]]) -> None:
        """Record experiences collected from external modules (for example, EnvSampler).

        Args:
            experiences (List[ExpElement]): List of experiences. Each ExpElement stores the complete information for a
                tick. Please refers to the definition of ExpElement for detailed explanation of ExpElement.
        """
        for env_idx, env_experience in enumerate(experiences):
            trainer_exp_pool = collections.defaultdict(list)
            for exp_element in env_experience:  # Dispatch experiences to trainers tick by tick.
                exp_dict = exp_element.split_contents_by_trainer(self._agent2trainer)
                for trainer_name, exp_elem in exp_dict.items():
                    trainer_exp_pool[trainer_name].append(exp_elem)

            for trainer_name, exp_elems in trainer_exp_pool.items():
                trainer = self._trainer_dict[trainer_name]
                trainer.record_multiple(env_idx, exp_elems)

    def load(self, path: str) -> List[str]:
        loaded = []
        for trainer_name, trainer in self._trainer_dict.items():
            trainer.load(path)
            loaded.append(trainer_name)
        return loaded

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for trainer_name, trainer in self._trainer_dict.items():
            trainer.save(path)

    def exit(self) -> None:
        if self._proxy_address:

            async def exit_all() -> Iterable:
                return await asyncio.gather(*[trainer.exit() for trainer in self._trainer_dict.values()])

            asyncio.run(exit_all())
