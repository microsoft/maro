# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Tuple

from maro.rl_v3.learning import ExpElement
from maro.rl_v3.policy import RLPolicy

from .trainer import AbsTrainer
from .utils import extract_trainer_name


class AbsTrainerManager(object, metaclass=ABCMeta):
    """
    Use TrainerManager to manage all policy trainers and handle the training process.
    """
    def __init__(self) -> None:
        super(AbsTrainerManager, self).__init__()

    def train(self) -> None:
        """
        Run a new round of training.
        """
        self._train_impl()

    @abstractmethod
    def _train_impl(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_policy_states(self) -> Dict[str, Dict[str, object]]:
        """
        Get policies' states.

        Returns:
            A double-deck dict with format: {trainer_name: {policy_name: policy_state}}
        """
        raise NotImplementedError

    @abstractmethod
    def record_experiences(self, experiences: List[ExpElement]) -> None:
        """
        Record experiences collected from external modules (for example, EnvSampler).

        Args:
            experiences (List[ExpElement]): List of experiences. Each ExpElement stores the complete information for a
                tick. Please refers to the definition of ExpElement for detailed explanation of ExpElement.
        """
        raise NotImplementedError


class SimpleTrainerManager(AbsTrainerManager):
    def __init__(
        self,
        policy_creator: Dict[str, Callable[[str], RLPolicy]],
        trainer_creator: Dict[str, Callable[[str], AbsTrainer]],
        agent2policy: Dict[str, str],  # {agent_name: policy_name}
        dispatcher_address: Tuple[str, int] = None
    ) -> None:
        """
        Simple trainer manager. Use this in centralized model.

        Args:
            policy_creator (Dict[str, Callable[[str], RLPolicy]]): Dict of functions to create policies.
            trainer_creator (Dict[str, Callable[[str], AbsTrainer]]): Dict of functions to create trainers.
            agent2policy (Dict[str, str]): Agent name to policy name mapping.
            dispatcher_address (Tuple[str, int]): The address of the dispatcher. This is used under only distributed
                model. Defaults to None.
        """
        super(SimpleTrainerManager, self).__init__()

        self._trainer_dict: Dict[str, AbsTrainer] = {}
        self._trainers: List[AbsTrainer] = []
        self._agent2policy = agent2policy
        for trainer_name, func in trainer_creator.items():
            trainer = func(trainer_name)
            if dispatcher_address is not None:
                trainer.set_dispatch_address(dispatcher_address)
            trainer.register_agent2policy(self._agent2policy)
            trainer.register_policy_creator(policy_creator)
            self._trainer_dict[trainer_name] = trainer
            self._trainers.append(trainer)

        asyncio.run(self._gather("build"))

    def _train_impl(self) -> None:
        asyncio.run(self._gather("train_step"))

    def get_policy_states(self) -> Dict[str, Dict[str, object]]:
        policy_state_list = asyncio.run(self._gather("get_policy_state"))
        return dict((trainer.name, policy_state) for trainer, policy_state in zip(self._trainers, policy_state_list))

    async def _gather(self, coroutine_name: str) -> list:
        return await asyncio.gather(*[getattr(trainer, coroutine_name)() for trainer in self._trainers])

    def record_experiences(self, experiences: List[ExpElement]) -> None:
        for exp_element in experiences:  # Dispatch experiences to trainers tick by tick.
            self._dispatch_experience(exp_element)

    def _dispatch_experience(self, exp_element: ExpElement) -> None:
        agent2trainer = {
            agent_name: extract_trainer_name(policy_name)
            for agent_name, policy_name in self._agent2policy.items()
        }
        exp_dict = exp_element.split_contents(agent2trainer)

        for trainer_name, exp_elem in exp_dict.items():
            trainer = self._trainer_dict[trainer_name]
            trainer.record(exp_elem)
