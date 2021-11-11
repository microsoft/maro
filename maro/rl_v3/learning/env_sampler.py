from abc import ABCMeta, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from maro.rl_v3.policy import RLPolicy
from maro.rl_v3.utils import ActionWithAux
from maro.simulator import Env


class AbsAgentWrapper(object, metaclass=ABCMeta):
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

    def choose_action_with_aux(self, state_by_agent: Dict[str, np.ndarray]) -> Dict[str, ActionWithAux]:
        with torch.no_grad():
            ret = self._choose_action_with_aux_impl(state_by_agent)
        return ret

    @abstractmethod
    def _choose_action_with_aux_impl(self, state_by_agent: Dict[str, np.ndarray]) -> Dict[str, ActionWithAux]:
        raise NotImplementedError

    @abstractmethod
    def explore(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def exploit(self) -> None:
        raise NotImplementedError


class SimpleAgentWrapper(AbsAgentWrapper):
    def __init__(
        self,
        policy_dict: Dict[str, RLPolicy],  # {policy_name: RLPolicy}
        agent2policy: Dict[str, str]  # {agent_name: policy_name}
    ) -> None:
        super(SimpleAgentWrapper, self).__init__(policy_dict=policy_dict, agent2policy=agent2policy)

    def _choose_action_with_aux_impl(self, state_by_agent: Dict[str, np.ndarray]) -> Dict[str, ActionWithAux]:
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
            policy.eval()
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
class AbsEnvSampler(object, metaclass=ABCMeta):
    def __init__(
        self,
        get_env_func: Callable[[], Env],
        #
        get_policy_func_dict: Dict[str, Callable[[str], RLPolicy]],
        agent2policy: Dict[str, str],  # {agent_name: policy_name}
        agent_wrapper_cls: Type[AbsAgentWrapper],
        reward_eval_delay: int = 0,
        get_test_env_func: Callable[[], Env] = None
    ) -> None:
        self._learn_env = get_env_func()
        self._test_env = get_test_env_func() if get_test_env_func is not None else self._learn_env
        self._env: Optional[Env] = None

        self._policy_dict: Dict[str, RLPolicy] = {
            policy_name: func(policy_name) for policy_name, func in get_policy_func_dict.items()
        }
        self._agent_wrapper = agent_wrapper_cls(self._policy_dict, agent2policy)
        self._agent2policy = agent2policy

        # Global state & agent state
        self._global_state: Optional[np.ndarray] = None
        self._agent_state_dict: Dict[str, np.ndarray] = {}

        self._trans_cache: Deque[CacheElement] = deque()
        self._reward_eval_delay = reward_eval_delay

        self._tracker = {}

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
        raise NotImplementedError

    @abstractmethod
    def _translate_to_env_action(self, action_with_aux_dict: Dict[str, ActionWithAux], event) -> Dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, env_action_dict: Dict[str, object], tick: int) -> Dict[str, float]:
        raise NotImplementedError

    def sample(  # TODO: check logic with qiuyang
        self,
        policy_state_dict: Dict[str, object] = None,
        num_steps: int = -1
    ) -> dict:
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
            self._global_state, self._agent_state_dict = (None, {}) if done \
                else self._get_global_and_agent_state(event)
            steps_to_go -= 1

        #
        tick_bound = self._env.tick - self._reward_eval_delay
        experiences = []
        while len(self._trans_cache) > 0 and self._trans_cache[0].tick <= tick_bound:
            cache_element = self._trans_cache.popleft()

            reward_dict = self._get_reward(cache_element.env_action_dict, cache_element.tick)
            self._post_step(cache_element, reward_dict)

            next_global_state = self._trans_cache[0].global_state if len(self._trans_cache) > 0 else self._global_state
            next_agent_state_dict = self._trans_cache[0].agent_state_dict if len(self._trans_cache) > 0 \
                else self._agent_state_dict
            terminal = not self._global_state and len(self._trans_cache) == 0

            experiences.append(ExpElement(
                tick=cache_element.tick,
                global_state=cache_element.global_state,
                agent_state_dict=cache_element.agent_state_dict,
                action_with_aux_dict=cache_element.action_with_aux_dict,
                env_action_dict=cache_element.env_action_dict,
                reward_dict=reward_dict,
                terminal=terminal,
                next_global_state=next_global_state,
                next_agent_state_dict=next_agent_state_dict
            ))
        return {
            "end_of_episode": not self._agent_state_dict,
            "experiences": experiences,
            "tracker": self._tracker
        }

    def test(self, policy_state_dict: Dict[str, object] = None) -> dict:
        self._env = self._test_env
        if policy_state_dict is not None:
            self._agent_wrapper.set_policy_states(policy_state_dict)

        self._agent_wrapper.exploit()
        self._env.reset()
        terminal = False
        _, event, _ = self._env.step(None)
        _, agent_state_dict = self._get_global_and_agent_state(event)
        while not terminal:
            action_with_aux_dict = self._agent_wrapper.choose_action_with_aux(agent_state_dict)
            env_action_dict = self._translate_to_env_action(action_with_aux_dict, event)
            _, event, terminal = self._env.step(list(env_action_dict.values()))
            if not terminal:
                _, agent_state_dict = self._get_global_and_agent_state(event)
        return self._tracker

    @abstractmethod
    def _post_step(
        self, cache_element: CacheElement, reward: Dict[str, float]
    ) -> None:
        raise NotImplementedError
