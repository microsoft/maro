from abc import abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple, Type

import numpy as np

from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.policy_learner import AbsLearner
from maro.rl_v3.policy_learner.abs_learner import SingleLearner
from maro.simulator import Env


class ActionWithAux:
    action: np.ndarray
    value: float = None
    logp: float = None


class AbsAgentWrapper(object):
    def __init__(
        self,
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        agent2policy: Dict[str, str]  # {agent_name: policy_name}
    ) -> None:
        self._policy_dict: Dict[str, RLPolicy] = {
            policy_name: func(policy_name) for policy_name, func in get_policy_func_dict.items()
        }
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


@dataclass
class CacheElement:
    tick: int
    global_state: np.ndarray
    agent_state_dict: Dict[str, np.ndarray]
    action_with_aux_dict: Dict[str, ActionWithAux]
    env_action_dict: Dict[str, object]


# TODO: event typehint
class AbsEnvSampler(object):
    def __init__(
        self,
        get_env_func: Callable[[], Env],
        #
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        get_learner_func_dict: Dict[str, Callable[[str], AbsLearner]],
        agent2policy: Dict[str, str],  # {agent_name: policy_name}
        policy2learner: Dict[str, str],  # {policy_name: learner_name}
        #
        agent_wrapper_cls: Type[AbsAgentWrapper],
        reward_eval_delay: int = 0
    ) -> None:
        self._learn_env = get_env_func()
        self._agent_wrapper = agent_wrapper_cls(get_policy_func_dict, agent2policy)

        self._learner_dict: Dict[str, AbsLearner] = {
            learner_name: func(learner_name) for learner_name, func in get_learner_func_dict.items()
        }
        self._agent2policy = agent2policy
        self._policy2learner = policy2learner

        self._env: Optional[Env] = None
        self._global_state: Optional[np.ndarray] = None
        self._agent_state_dict: Dict[str, np.ndarray] = {}

        self._trans_cache: Deque[CacheElement] = deque()
        self._reward_eval_delay = reward_eval_delay

    @abstractmethod
    def _get_global_and_agent_state(self, event, tick: int = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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
    ):  # TODO: return typehint
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

        # Dispatch experience to learners
        tick_bound = self._env.tick - self._reward_eval_delay
        while len(self._trans_cache) > 0 and self._trans_cache[0].tick <= tick_bound:
            cache_element = self._trans_cache.popleft()
            reward_dict = self._get_reward(cache_element.env_action_dict, cache_element.tick)
            self._post_step(cache_element, reward_dict)

            next_global_state = self._trans_cache[0].global_state if len(self._trans_cache) > 0 else self._global_state
            next_agent_state_dict = self._trans_cache[0].agent_state_dict if len(self._trans_cache) > 0 \
                else self._agent_state_dict
            self._dispatch_experience(
                cache_element=cache_element,
                reward_dict=reward_dict,
                terminal=not self._global_state and len(self._trans_cache) == 0,
                next_global_state=next_global_state,
                next_agent_state_dict=next_agent_state_dict
            )

    def _dispatch_experience(
        self,
        cache_element: CacheElement,
        reward_dict: Dict[str, float],
        terminal: bool,
        next_global_state: np.ndarray = None,
        next_agent_state_dict: Dict[str, np.ndarray] = None
    ) -> None:
        learner_buffer = defaultdict(List[Tuple[str, np.ndarray, ActionWithAux, float]])
        for agent_name, agent_state in cache_element.agent_state_dict.items():
            policy_name = self._agent2policy[agent_name]
            learner_name = self._policy2learner[policy_name]

            action_with_aux = cache_element.action_with_aux_dict[agent_name]
            reward = reward_dict[agent_name]

            learner_buffer[learner_name].append((policy_name, agent_state, action_with_aux, reward))

        for learner_name, exps in learner_buffer.items():
            learner = self._learner_dict[learner_name]
            if isinstance(learner, SingleLearner):
                assert len(exps) == 1
                policy_name, agent_state, action_with_aux, reward = exps[0]
                learner.record(
                    policy_name=policy_name,
                    state=agent_state,
                    action=action_with_aux.action,
                    reward=reward,
                    terminal=terminal,
                    next_state=next_global_state,
                    value=action_with_aux.value,
                    logp=action_with_aux.logp
                )
            else:  # TODO: MultiLearner case. To be implemented.
                pass

    @abstractmethod
    def _post_step(
        self, cache_element: CacheElement, reward: Dict[str, float]
    ) -> None:
        pass
