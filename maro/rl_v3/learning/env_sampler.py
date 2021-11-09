from abc import abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple, Type

import numpy as np

from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.policy_trainer import AbsTrainer, SingleTrainer
from maro.rl_v3.utils import ActionWithAux, TransitionBatch
from maro.simulator import Env


class AbsAgentWrapper(object):
    def __init__(
        self,
        policy_dict: Dict[str, RLPolicy],  # {policy_name: RLPolicy}
        agent2policy: Dict[str, str]  # {agent_name: policy_name}
    ) -> None:
        self._policy_dict = policy_dict
        self._agent2policy = agent2policy

    def set_policy_states(self, policy_state_dict: Dict[str, object]) -> None:
        for policy_name, policy_state in policy_state_dict.items():
            policy = self._policy_dict[policy_name]
            policy.set_policy_state(policy_state)

    @abstractmethod
    def choose_action_with_aux(self, state_by_agent: Dict[str, np.ndarray]) -> Dict[str, ActionWithAux]:
        pass

    @abstractmethod
    def explore(self) -> None:
        pass

    @abstractmethod
    def exploit(self) -> None:
        pass


class SimpleAgentWrapper(AbsAgentWrapper):
    def __init__(
        self,
        policy_dict: Dict[str, RLPolicy],  # {policy_name: RLPolicy}
        agent2policy: Dict[str, str]  # {agent_name: policy_name}
    ) -> None:
        super(SimpleAgentWrapper, self).__init__(policy_dict=policy_dict, agent2policy=agent2policy)

    def choose_action_with_aux(self, state_by_agent: Dict[str, np.ndarray]) -> Dict[str, ActionWithAux]:
        # Aggregate states by policy
        states_by_policy = defaultdict(list)  # {str: list of np.ndarray}
        agents_by_policy = defaultdict(list)  # {str: list of str}
        for agent_name, state in state_by_agent.items():
            policy_name = self._agent2policy[agent_name]
            states_by_policy[policy_name].append(state)
            agents_by_policy[policy_name].append(agent_name)

        action_with_aux_dict = {}
        for policy_name in agents_by_policy:
            policy = self._policy_dict[policy_name]
            states = np.vstack(states_by_policy[policy_name])  # np.ndarray
            action_with_aux_dict.update(zip(
                agents_by_policy[policy_name],  # list of str (agent name)
                policy.get_actions_with_aux(states)  # list of action_with_aux
            ))
        return action_with_aux_dict

    def explore(self) -> None:
        for policy in self._policy_dict.values():
            policy.explore()

    def exploit(self) -> None:
        for policy in self._policy_dict.values():
            policy.exploit()


@dataclass
class CacheElement:
    tick: int
    global_state: np.ndarray
    agent_state_dict: Dict[str, np.ndarray]
    action_with_aux_dict: Dict[str, ActionWithAux]
    env_action_dict: Dict[str, object]


@dataclass  # TODO: check if all fields are needed
class ExpElement(CacheElement):
    reward_dict: Dict[str, float]
    terminal: bool
    next_global_state: np.ndarray = None
    next_agent_state_dict: Dict[str, np.ndarray] = None


# TODO: event typehint
class AbsEnvSampler(object):
    def __init__(
        self,
        get_env_func: Callable[[], Env],
        #
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        get_trainer_func_dict: Dict[str, Callable[[str], AbsTrainer]],
        agent2policy: Dict[str, str],  # {agent_name: policy_name}
        policy2trainer: Dict[str, str],  # {policy_name: trainer_name}
        #
        agent_wrapper_cls: Type[AbsAgentWrapper],
        reward_eval_delay: int = 0,
        #
        return_experiences: bool = False
    ) -> None:
        self._learn_env = get_env_func()

        self._policy_dict: Dict[str, RLPolicy] = {
            policy_name: func(policy_name) for policy_name, func in get_policy_func_dict.items()
        }
        self._agent_wrapper = agent_wrapper_cls(self._policy_dict, agent2policy)
        self._agent2policy = agent2policy
        self._env: Optional[Env] = None
        self._global_state: Optional[np.ndarray] = None
        self._agent_state_dict: Dict[str, np.ndarray] = {}

        self._trans_cache: Deque[CacheElement] = deque()
        self._reward_eval_delay = reward_eval_delay
        self._return_experiences = return_experiences

        self._tracker = {}

        if not return_experiences:  # Build and init trainers inside env_sampler
            self._trainer_dict: Dict[str, AbsTrainer] = {
                trainer_name: func(trainer_name) for trainer_name, func in get_trainer_func_dict.items()
            }
            self._policy2trainer = policy2trainer
            for policy_name, trainer_name in self._policy2trainer.items():
                policy = self._policy_dict[policy_name]
                trainer = self._trainer_dict[trainer_name]
                assert isinstance(trainer, SingleTrainer)  # TODO: extend for multi trainer
                trainer.register_policy(policy)  # TODO: extend for multi trainer

    @abstractmethod
    def _get_global_and_agent_state(
        self, event, tick: int = None
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Get global state and dict of agent states.

        Args:
            event: Event.
            tick (int): Tick.

        Returns:
            Global state: np.ndarray
            Dict of agent states: Dict[str, np.ndarray]
        """
        pass

    @abstractmethod
    def _translate_to_env_action(self, action_with_aux_dict: Dict[str, ActionWithAux], event) -> Dict[str, object]:
        pass

    @abstractmethod
    def _get_reward(self, env_action_dict: Dict[str, object], tick: int) -> Dict[str, float]:
        pass

    def sample(  # TODO: check logic with qiuyang
        self,
        policy_state_dict: Dict[str, object] = None,
        num_steps: int = -1
    ) -> Optional[List[ExpElement]]:  # TODO: return typehint
        # Init the env
        self._env = self._learn_env
        if not self._agent_state_dict:
            self._env.reset()
            self._trans_cache.clear()
            _, event, _ = self._env.step(None)
            self._global_state, self._agent_state_dict = self._get_global_and_agent_state(event)
        else:
            event = None

        # Update policy state if necessary
        if policy_state_dict is not None:
            self._agent_wrapper.set_policy_states(policy_state_dict)

        # Collect experience
        self._agent_wrapper.explore()
        steps_to_go = float("inf") if num_steps == -1 else num_steps
        while self._agent_state_dict and steps_to_go > 0:
            # Get agent actions and translate them to env actions
            action_with_aux_dict = self._agent_wrapper.choose_action_with_aux(self._agent_state_dict)
            env_action_dict = self._translate_to_env_action(action_with_aux_dict, event)
            # Store experiences in the cache
            self._trans_cache.append(
                CacheElement(
                    tick=self._env.tick,
                    global_state=self._global_state,
                    agent_state_dict=dict(self._agent_state_dict),
                    action_with_aux_dict=action_with_aux_dict,
                    env_action_dict=env_action_dict
                )
            )
            # Update env and get new states (global & agent)
            _, event, done = self._env.step(list(env_action_dict.values()))
            self._global_state, self._agent_state_dict = None, {} if done else self._get_global_and_agent_state(event)
            steps_to_go -= 1

        # Dispatch experience to trainers
        tick_bound = self._env.tick - self._reward_eval_delay
        experiences = []
        while len(self._trans_cache) > 0 and self._trans_cache[0].tick <= tick_bound:
            cache_element = self._trans_cache.popleft()
            reward_dict = self._get_reward(cache_element.env_action_dict, cache_element.tick)
            self._post_step(cache_element, reward_dict)

            next_global_state = self._trans_cache[0].global_state if len(self._trans_cache) > 0 else self._global_state
            next_agent_state_dict = self._trans_cache[0].agent_state_dict if len(self._trans_cache) > 0 \
                else self._agent_state_dict

            experiences.append(ExpElement(
                tick=cache_element.tick,
                global_state=cache_element.global_state,
                agent_state_dict=cache_element.agent_state_dict,
                action_with_aux_dict=cache_element.action_with_aux_dict,
                env_action_dict=cache_element.env_action_dict,
                reward_dict=reward_dict,
                terminal=not self._global_state and len(self._trans_cache) == 0,
                next_global_state=next_global_state,
                next_agent_state_dict=next_agent_state_dict
            ))

        if self._return_experiences:
            return experiences
        else:
            for exp_element in experiences:
                self._dispatch_experience(exp_element)

    def _dispatch_experience(
        self,
        exp_element: ExpElement
    ) -> None:
        agent_state_dict = exp_element.agent_state_dict
        action_with_aux_dict = exp_element.action_with_aux_dict
        reward_dict = exp_element.reward_dict
        terminal = exp_element.terminal
        next_global_state = exp_element.next_global_state

        trainer_buffer = defaultdict(List[Tuple[str, np.ndarray, ActionWithAux, float]])
        for agent_name, agent_state in agent_state_dict.items():
            policy_name = self._agent2policy[agent_name]
            trainer_name = self._policy2trainer[policy_name]

            action_with_aux = action_with_aux_dict[agent_name]
            reward = reward_dict[agent_name]

            trainer_buffer[trainer_name].append((policy_name, agent_state, action_with_aux, reward))

        for trainer_name, exps in trainer_buffer.items():
            trainer = self._trainer_dict[trainer_name]
            if isinstance(trainer, SingleTrainer):
                assert len(exps) == 1
                policy_name, agent_state, action_with_aux, reward = exps[0]
                batch = TransitionBatch(
                    states=np.expand_dims(agent_state, axis=0),
                    actions=np.expand_dims(action_with_aux.action, axis=0),
                    rewards=np.array([reward]),
                    terminals=np.array([terminal]),
                    next_states=None if next_global_state is None else np.expand_dims(next_global_state, axis=0),
                    values=None if action_with_aux.value is None else np.array([action_with_aux.value]),
                    logps=None if action_with_aux.logp is None else np.array([action_with_aux.logp]),
                )
                trainer.record(policy_name=policy_name, transition_batch=batch)
            else:  # TODO: MultiLearner case. To be implemented.
                pass

    @abstractmethod
    def _post_step(
        self, cache_element: CacheElement, reward: Dict[str, float]
    ) -> None:
        pass
