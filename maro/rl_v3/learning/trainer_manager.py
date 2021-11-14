from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np

from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.policy_trainer import AbsTrainer, MultiTrainer, SingleTrainer
from maro.rl_v3.utils import ActionWithAux, MultiTransitionBatch, TransitionBatch
from .env_sampler import ExpElement


class AbsTrainerManager(object, metaclass=ABCMeta):
    """
    Use TrainerManager to manage all policy trainers and handle the training process.
    """
    def __init__(self) -> None:
        super(AbsTrainerManager, self).__init__()

    @abstractmethod
    def train(self) -> None:
        """
        Run a new round of training.
        """
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
        get_trainer_func_dict: Dict[str, Callable[[str], AbsTrainer]],
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        agent2policy: Dict[str, str],  # {agent_name: policy_name}
        policy2trainer: Dict[str, str],  # {policy_name: trainer_name}
    ) -> None:
        """
        Simple trainer manager. Use this in centralized model.

        Args:
            get_trainer_func_dict (Dict[str, Callable[[str], AbsTrainer]]): Dict of functions used to create trainers.
            get_policy_func_dict (Dict[str, Callable[[str], RLPolicy]]): Dict of functions used to create policies.
            agent2policy (Dict[str, str]): Agent name to policy name mapping.
            policy2trainer (Dict[str, str]): Policy name to trainer name mapping.
        """

        super(SimpleTrainerManager, self).__init__()

        self._trainer_dict: Dict[str, AbsTrainer] = {name: func(name) for name, func in get_trainer_func_dict.items()}
        self._trainers: List[AbsTrainer] = list(self._trainer_dict.values())

        self._policy_dict: Dict[str, RLPolicy] = {name: func(name) for name, func in get_policy_func_dict.items()}

        self._agent2policy = agent2policy
        self._policy2trainer = policy2trainer

        # register policies
        trainer_policies = defaultdict(list)
        for policy_name, trainer_name in self._policy2trainer.items():
            policy = self._policy_dict[policy_name]
            trainer_policies[trainer_name].append(policy)
        for trainer_name, trainer in self._trainer_dict.items():
            policies = trainer_policies[trainer_name]
            if isinstance(trainer, SingleTrainer):
                assert len(policies) == 1
                trainer.register_policy(policies[0])
            elif isinstance(trainer, MultiTrainer):
                trainer.register_policies(policies)
            else:
                raise ValueError

    def train(self) -> None:
        for trainer in self._trainers:
            trainer.train_step()

    def get_policy_states(self) -> Dict[str, Dict[str, object]]:
        return {trainer.name: trainer.get_policy_state_dict() for trainer in self._trainers}

    def record_experiences(self, experiences: List[ExpElement]) -> None:
        for exp_element in experiences:  # Dispatch experiences to trainers tick by tick.
            self._dispatch_experience(exp_element)

    def _dispatch_experience(self, exp_element: ExpElement) -> None:
        state = exp_element.global_state
        agent_state_dict = exp_element.agent_state_dict
        action_with_aux_dict = exp_element.action_with_aux_dict
        reward_dict = exp_element.reward_dict
        terminal_dict = exp_element.terminal_dict
        next_state = exp_element.next_global_state
        next_agent_state_dict = exp_element.next_agent_state_dict

        # Aggregate experiences by trainer
        trainer_buffer = defaultdict(list)
        for agent_name, agent_state in agent_state_dict.items():
            policy_name = self._agent2policy[agent_name]
            trainer_name = self._policy2trainer[policy_name]

            action_with_aux = action_with_aux_dict[agent_name]
            reward = reward_dict[agent_name]

            trainer_buffer[trainer_name].append((
                policy_name, agent_state, action_with_aux, reward, next_agent_state_dict.get(agent_name, None),
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
                action_with_aux: ActionWithAux = exps[0][2]
                reward: float = exps[0][3]
                next_agent_state: np.ndarray = exps[0][4]
                terminal: bool = exps[0][5]

                batch = TransitionBatch(
                    policy_name=policy_name,
                    states=np.expand_dims(agent_state, axis=0),
                    actions=np.expand_dims(action_with_aux.action, axis=0),
                    rewards=np.array([reward]),
                    terminals=np.array([terminal]),
                    next_states=None if next_agent_state is None else np.expand_dims(next_agent_state, axis=0),
                    values=None if action_with_aux.value is None else np.array([action_with_aux.value]),
                    logps=None if action_with_aux.logp is None else np.array([action_with_aux.logp]),
                )
                trainer.record(policy_name=policy_name, transition_batch=batch)
            elif isinstance(trainer, MultiTrainer):
                policy_names: List[str] = []
                actions: List[np.ndarray] = []
                rewards: List[np.ndarray] = []
                terminals: List[bool] = []
                agent_states: List[np.ndarray] = []
                next_agent_states: List[np.ndarray] = []
                values: List[np.ndarray] = []
                logps: List[np.ndarray] = []

                next_agent_states_flag = True
                values_flag = True
                logps_flag = True

                for exp in exps:
                    policy_name: str = exp[0]
                    agent_state: np.ndarray = exp[1]
                    action_with_aux: ActionWithAux = exp[2]
                    reward: float = exp[3]
                    next_agent_state: np.ndarray = exp[4]
                    terminal: bool = exp[5]

                    policy_names.append(policy_name)
                    actions.append(np.expand_dims(action_with_aux.action, axis=0))
                    rewards.append(np.array([reward]))
                    terminals.append(terminal)
                    agent_states.append(np.expand_dims(agent_state, axis=0))

                    if not next_agent_states_flag or next_agent_state is None:
                        next_agent_states_flag = False
                    else:
                        next_agent_states.append(np.expand_dims(next_agent_state, axis=0))

                    if not values_flag or action_with_aux.value is None:
                        values_flag = False
                    else:
                        values.append(np.array([action_with_aux.value]))

                    if not logps_flag or action_with_aux.logp is None:
                        logps_flag = False
                    else:
                        logps.append(np.array([action_with_aux.logp]))

                batch = MultiTransitionBatch(
                    policy_names=policy_names,
                    states=np.expand_dims(state, axis=0),
                    actions=actions,
                    rewards=rewards,
                    terminals=np.array(terminals),
                    agent_states=agent_states,
                    next_states=np.expand_dims(next_state, axis=0),
                    next_agent_states=None if not next_agent_states_flag else next_agent_states,
                    values=None if not values_flag else values,
                    logps=None if not logps_flag else logps
                )
                trainer.record(batch)
            else:
                raise ValueError
