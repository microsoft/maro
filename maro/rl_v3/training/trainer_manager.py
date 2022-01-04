# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np

from maro.rl_v3.learning import ExpElement
from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.utils import MultiTransitionBatch, TransitionBatch

from .trainer import AbsTrainer, MultiTrainer, SingleTrainer
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
    async def _train_impl(self) -> None:
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
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        get_trainer_func_dict: Dict[str, Callable[[str], AbsTrainer]],
        agent2policy: Dict[str, str],  # {agent_name: policy_name}
        dispatcher_address: Tuple[str, int] = None
    ) -> None:
        """
        Simple trainer manager. Use this in centralized model.

        Args:
            get_policy_func_dict (Dict[str, Callable[[str], RLPolicy]]): Dict of functions used to create policies.
            get_trainer_func_dict (Dict[str, Callable[[str], AbsTrainer]]): Dict of functions used to create trainers.
            agent2policy (Dict[str, str]): Agent name to policy name mapping.
            dispatcher_address (Tuple[str, int]): The address of the dispatcher. This is used under only distributed
                model. Defaults to None.
        """
        super(SimpleTrainerManager, self).__init__()

        self._trainer_dict: Dict[str, AbsTrainer] = {}
        self._trainers: List[AbsTrainer] = []
        for trainer_name, func in get_trainer_func_dict.items():
            trainer = func(trainer_name)
            if dispatcher_address is not None:
                trainer.set_dispatch_address(dispatcher_address)
            trainer.register_get_policy_func_dict(get_policy_func_dict)
            trainer.build()

            self._trainer_dict[trainer_name] = trainer
            self._trainers.append(trainer)

        self._policy_dict: Dict[str, RLPolicy] = {name: func(name) for name, func in get_policy_func_dict.items()}
        self._agent2policy = agent2policy

    def train(self) -> None:
        asyncio.run(self._train_impl())

    async def _train_impl(self) -> None:
        await asyncio.gather(*[trainer.train_step() for trainer in self._trainers])

    def get_policy_states(self) -> Dict[str, Dict[str, object]]:
        return {trainer.name: trainer.get_policy_state_dict() for trainer in self._trainers}

    def record_experiences(self, experiences: List[ExpElement]) -> None:
        for exp_element in experiences:  # Dispatch experiences to trainers tick by tick.
            self._dispatch_experience(exp_element)

    def _dispatch_experience(self, exp_element: ExpElement) -> None:
        state = exp_element.state
        agent_state_dict = exp_element.agent_state_dict
        action_dict = exp_element.action_dict
        reward_dict = exp_element.reward_dict
        terminal_dict = exp_element.terminal_dict
        next_state = exp_element.next_state
        next_agent_state_dict = exp_element.next_agent_state_dict

        if next_state is None:
            next_state = state

        # Aggregate experiences by trainer
        trainer_buffer = defaultdict(list)
        for agent_name, agent_state in agent_state_dict.items():
            policy_name = self._agent2policy[agent_name]
            trainer_name = extract_trainer_name(policy_name)
            action = action_dict[agent_name]
            reward = reward_dict[agent_name]
            trainer_buffer[trainer_name].append((
                policy_name,
                agent_state,
                action,
                reward,
                next_agent_state_dict[agent_name] if agent_name in next_agent_state_dict else agent_state,
                terminal_dict[agent_name]
            ))

        for trainer_name, exps in trainer_buffer.items():
            if trainer_name not in self._trainer_dict:
                continue
            trainer = self._trainer_dict[trainer_name]
            if isinstance(trainer, SingleTrainer):
                assert len(exps) == 1, f"SingleTrainer must has exactly one policy. Currently, it has {len(exps)}."

                policy_name: str = exps[0][0]
                agent_state: np.ndarray = exps[0][1]
                action: np.ndarray = exps[0][2]
                reward: float = exps[0][3]
                next_agent_state: np.ndarray = exps[0][4]
                terminal: bool = exps[0][5]

                batch = TransitionBatch(
                    policy_name=policy_name,
                    states=np.expand_dims(agent_state, axis=0),
                    actions=np.expand_dims(action, axis=0),
                    rewards=np.array([reward]),
                    next_states=np.expand_dims(next_agent_state, axis=0),
                    terminals=np.array([terminal])
                )
                trainer.record(transition_batch=batch)
            elif isinstance(trainer, MultiTrainer):
                policy_names: List[str] = []
                actions: List[np.ndarray] = []
                rewards: List[np.ndarray] = []
                terminals: List[bool] = []
                agent_states: List[np.ndarray] = []
                next_agent_states: List[np.ndarray] = []

                for exp in exps:
                    policy_name: str = exp[0]
                    agent_state: np.ndarray = exp[1]
                    action: np.ndarray = exp[2]
                    reward: float = exp[3]
                    next_agent_state: np.ndarray = exp[4]
                    terminal: bool = exp[5]

                    policy_names.append(policy_name)
                    actions.append(np.expand_dims(action, axis=0))
                    rewards.append(np.array([reward]))
                    terminals.append(terminal)
                    agent_states.append(np.expand_dims(agent_state, axis=0))
                    next_agent_states.append(np.expand_dims(next_agent_state, axis=0))

                batch = MultiTransitionBatch(
                    policy_names=policy_names,
                    states=np.expand_dims(state, axis=0),
                    actions=actions,
                    rewards=rewards,
                    next_states=np.expand_dims(next_state, axis=0),
                    agent_states=agent_states,
                    next_agent_states=next_agent_states,
                    terminals=np.array(terminals)
                )
                trainer.record(batch)
            else:
                raise ValueError
