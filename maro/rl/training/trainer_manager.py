# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from itertools import chain
from typing import Callable, Dict, Iterable, List, Tuple

from maro.rl.policy import RLPolicy
from maro.rl.rollout import ExpElement

from .trainer import AbsTrainer
from .utils import extract_trainer_name


class TrainerManager(object):
    def __init__(
        self,
        policy_creator: Dict[str, Callable[[str], RLPolicy]],
        trainer_creator: Dict[str, Callable[[str], AbsTrainer]],
        agent2policy: Dict[str, str],  # {agent_name: policy_name}
        dispatcher_address: Tuple[str, int] = None
    ) -> None:
        """
        Trainer manager.

        Args:
            policy_creator (Dict[str, Callable[[str], RLPolicy]]): Dict of functions to create policies.
            trainer_creator (Dict[str, Callable[[str], AbsTrainer]]): Dict of functions to create trainers.
            agent2policy (Dict[str, str]): Agent name to policy name mapping.
            dispatcher_address (Tuple[str, int]): The address of the dispatcher. This is used under only distributed
                model. Defaults to None.
        """
        super(TrainerManager, self).__init__()

        self._trainer_dict: Dict[str, AbsTrainer] = {}
        self._trainers: List[AbsTrainer] = []
        self._agent2policy = agent2policy
        self._dispatcher_address = dispatcher_address
        for trainer_name, func in trainer_creator.items():
            trainer = func(trainer_name)
            if self._dispatcher_address:
                trainer.set_dispatch_address(self._dispatcher_address)
            trainer.register_agent2policy(self._agent2policy)
            trainer.register_policy_creator(policy_creator)
            trainer.build()
            self._trainer_dict[trainer_name] = trainer
            self._trainers.append(trainer)

        self._agent2trainer = {
            agent_name: extract_trainer_name(policy_name)
            for agent_name, policy_name in self._agent2policy.items()
        }

    def train(self) -> None:
        if self._dispatcher_address:
            async def train_step() -> Iterable:
                return await asyncio.gather(*[trainer.train_as_task() for trainer in self._trainers])
            asyncio.run(train_step())
        else:
            for trainer in self._trainers:
                trainer.train()

    def get_policy_state(self) -> Dict[str, Dict[str, object]]:
        """Get policies' states.

        Returns:
            A double-deck dict with format: {trainer_name: {policy_name: policy_state}}
        """
        return dict(chain(*[trainer.get_policy_state().items() for trainer in self._trainers]))

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