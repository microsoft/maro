# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import os
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Tuple

from maro.rl.policy import AbsPolicy
from maro.rl.rollout import ExpElement
from maro.rl.training import SingleAgentTrainer
from maro.utils import LoggerV2
from maro.utils.exception.rl_toolkit_exception import MissingTrainer

from .trainer import AbsTrainer, MultiAgentTrainer
from .utils import extract_trainer_name, get_trainer_state_path


class TrainingManager(object):
    """
    Training manager. Manage and schedule all trainers to train policies.

    Args:
        policy_creator (Dict[str, Callable[[str], AbsPolicy]]): Dict of functions to create policies.
        trainer_creator (Dict[str, Callable[[str], AbsTrainer]]): Dict of functions to create trainers.
        agent2policy (Dict[Any, str]): Agent name to policy name mapping.
        device_mapping (Dict[str, str], default={}): User-defined device mapping from policy name to pytorch
            device name.
        proxy_address (Tuple[str, int], default=None): Address of the training proxy. If it is not None,
            it is registered to all trainers, which in turn create `RemoteOps` for distributed training.
        logger (LoggerV2, default=None): A logger for logging key events.
    """

    def __init__(
        self,
        policy_creator: Dict[str, Callable[[str], AbsPolicy]],
        trainer_creator: Dict[str, Callable[[str], AbsTrainer]],
        agent2policy: Dict[Any, str],  # {agent_name: policy_name}
        device_mapping: Dict[str, str] = None,
        proxy_address: Tuple[str, int] = None,
        logger: LoggerV2 = None,
    ) -> None:
        super(TrainingManager, self).__init__()

        self._trainer_dict: Dict[str, AbsTrainer] = {}
        self._agent2policy = agent2policy
        self._proxy_address = proxy_address
        for trainer_name, func in trainer_creator.items():
            trainer = func(trainer_name)
            if self._proxy_address:
                trainer.set_proxy_address(self._proxy_address)
            trainer.register_agent2policy(self._agent2policy)
            trainer.register_policy_creator(policy_creator)
            trainer.register_logger(logger)
            trainer.build()  # `build()` must be called after `register_policy_creator()`
            self._trainer_dict[trainer_name] = trainer

        # User-defined allocation of compute devices, i.e., GPU's to the trainer ops
        if device_mapping is not None:
            for policy_name, device_name in device_mapping.items():
                trainer = self._trainer_dict[extract_trainer_name(policy_name)]

                if isinstance(trainer, SingleAgentTrainer):
                    ops = trainer.ops
                else:
                    assert isinstance(trainer, MultiAgentTrainer)
                    ops = trainer.ops_dict[policy_name]
                ops.to_device(device_name)

        self._agent2trainer: Dict[Any, str] = {}
        for agent_name, policy_name in self._agent2policy.items():
            trainer_name = extract_trainer_name(policy_name)
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

    def get_policy_state(self) -> Dict[str, Dict[str, object]]:
        """Get policies' states.

        Returns:
            A double-deck dict with format: {trainer_name: {policy_name: policy_state}}
        """
        return dict(chain(*[trainer.get_policy_state().items() for trainer in self._trainer_dict.values()]))

    def record_experiences(self, experiences: List[List[ExpElement]]) -> None:
        """Record experiences collected from external modules (for example, EnvSampler).

        Args:
            experiences (List[ExpElement]): List of experiences. Each ExpElement stores the complete information for a
                tick. Please refers to the definition of ExpElement for detailed explanation of ExpElement.
        """
        for env_idx, env_experience in enumerate(experiences):
            for exp_element in env_experience:  # Dispatch experiences to trainers tick by tick.
                exp_dict = exp_element.split_contents(self._agent2trainer)
                for trainer_name, exp_elem in exp_dict.items():
                    trainer = self._trainer_dict[trainer_name]
                    trainer.record(env_idx, exp_elem)

    def load(self, path: str) -> List[str]:
        loaded = []
        for trainer_name, trainer in self._trainer_dict.items():
            pth = get_trainer_state_path(path, trainer_name)
            if os.path.isfile(pth):
                trainer.load(pth)
                loaded.append(trainer_name)

        return loaded

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for trainer_name, trainer in self._trainer_dict.items():
            trainer.save(get_trainer_state_path(path, trainer_name))

    def exit(self) -> None:
        if self._proxy_address:
            async def exit_all() -> Iterable:
                return await asyncio.gather(*[trainer.exit() for trainer in self._trainer_dict.values()])

            asyncio.run(exit_all())
