from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np

from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.policy_trainer import AbsTrainer, MultiTrainer, SingleTrainer
from maro.rl_v3.utils import MultiTransitionBatch, TransitionBatch

from .env_sampler import PolicyExpElement


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
        self.explore()
        self._train_impl()

    @abstractmethod
    def _train_impl(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def explore(self) -> None:
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
    def record_experiences(self, experiences: List[PolicyExpElement]) -> None:
        """
        Record experiences collected from external modules (for example, EnvSampler).

        Args:
            experiences (List[PolicyExpElement]): List of experiences. Each PolicyExpElement stores the complete
            information for a tick.
        """
        raise NotImplementedError


class SimpleTrainerManager(AbsTrainerManager):
    def __init__(
        self,
        get_trainer_func_dict: Dict[str, Callable[[str], AbsTrainer]],
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        policy2trainer: Dict[str, str],  # {policy_name: trainer_name}
    ) -> None:
        """
        Simple trainer manager. Use this in centralized model.

        Args:
            get_trainer_func_dict (Dict[str, Callable[[str], AbsTrainer]]): Dict of functions used to create trainers.
            get_policy_func_dict (Dict[str, Callable[[str], RLPolicy]]): Dict of functions used to create policies.
            policy2trainer (Dict[str, str]): Policy name to trainer name mapping.
        """

        super(SimpleTrainerManager, self).__init__()

        self._trainer_dict: Dict[str, AbsTrainer] = {name: func(name) for name, func in get_trainer_func_dict.items()}
        self._trainers: List[AbsTrainer] = list(self._trainer_dict.values())

        self._policy_dict: Dict[str, RLPolicy] = {name: func(name) for name, func in get_policy_func_dict.items()}

        self._policy2trainer = policy2trainer

        self._trainer2policies = defaultdict(list)

        # register policies
        trainer_policies = defaultdict(list)
        for policy_name, trainer_name in self._policy2trainer.items():
            policy = self._policy_dict[policy_name]
            trainer_policies[trainer_name].append(policy)
            self._trainer2policies[trainer_name].append(policy_name)
        for trainer_name, trainer in self._trainer_dict.items():
            policies = trainer_policies[trainer_name]
            if isinstance(trainer, SingleTrainer):
                assert len(policies) == 1
                trainer.register_policy(policies[0])
            elif isinstance(trainer, MultiTrainer):
                trainer.register_policies(policies)
            else:
                raise ValueError

    def _train_impl(self) -> None:
        for trainer in self._trainers:
            trainer.train_step()

    def explore(self) -> None:
        for policy in self._policy_dict.values():
            policy.explore()

    def get_policy_states(self) -> Dict[str, Dict[str, object]]:
        return {trainer.name: trainer.get_policy_state_dict() for trainer in self._trainers}

    def record_experiences(self, experiences: List[PolicyExpElement]) -> None:
        for exp_element in experiences:  # Dispatch experiences to trainers tick by tick.
            self._dispatch_experience(exp_element)

    def _dispatch_experience(self, exp_element: PolicyExpElement) -> None:
        policy_name_set = set(exp_element.policy_state_dict.keys())
        trainer_set = set(self._policy2trainer[policy_name] for policy_name in policy_name_set)

        for trainer_name in trainer_set:
            policy_names = self._trainer2policies[trainer_name]
            trainer = self._trainer_dict[trainer_name]

            if isinstance(trainer, SingleTrainer):
                assert len(policy_names) == 1
                policy_name = policy_names[0]

                batch = TransitionBatch(
                    policy_name=policy_name,
                    states=np.vstack(exp_element.policy_state_dict[policy_name]),
                    actions=np.vstack(exp_element.action_dict[policy_name]),
                    rewards=np.array(exp_element.reward_dict[policy_name]),
                    next_states=np.vstack(exp_element.next_policy_state_dict[policy_name]),
                    terminals=np.array(exp_element.terminal_dict[policy_name])
                )
                trainer.record(batch)
            elif isinstance(trainer, MultiTrainer):
                assert all(policy_name in policy_name_set for policy_name in policy_names)

                actions = []
                rewards = []
                policy_states = []
                next_policy_states = []
                terminals = []
                for policy_name in policy_names:
                    # TODO: currently we only allow 1-1 mapping between agents and polices under MARL scenarios
                    assert len(exp_element.action_dict[policy_name]) == 1
                    assert len(exp_element.reward_dict[policy_name]) == 1
                    assert len(exp_element.terminal_dict[policy_name]) == 1
                    assert len(exp_element.policy_state_dict[policy_name]) == 1
                    assert len(exp_element.next_policy_state_dict[policy_name]) == 1

                    actions.append(np.expand_dims(exp_element.action_dict[policy_name][0], axis=0))
                    rewards.append(np.array([exp_element.reward_dict[policy_name][0]]))
                    terminals.append(exp_element.terminal_dict[policy_name][0])
                    policy_states.append(np.expand_dims(exp_element.policy_state_dict[policy_name][0], axis=0))
                    next_policy_states.append(
                        np.expand_dims(exp_element.next_policy_state_dict[policy_name][0], axis=0)
                    )

                batch = MultiTransitionBatch(
                    policy_names=policy_names,
                    states=np.expand_dims(exp_element.state, axis=0),
                    actions=actions,
                    rewards=rewards,
                    next_states=np.expand_dims(exp_element.next_state, axis=0),
                    policy_states=policy_states,
                    next_policy_states=next_policy_states,
                    terminals=np.array(terminals)
                )
                trainer.record(batch)
            else:
                raise ValueError
