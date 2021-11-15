from abc import ABCMeta, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from maro.rl_v3.policy import RLPolicy
from maro.simulator import Env


class AbsAgentWrapper(object, metaclass=ABCMeta):
    """
    Agent wrapper is used to manager agents & policies during experiences collection.
    """
    def __init__(
        self,
        policy_dict: Dict[str, RLPolicy],  # {policy_name: RLPolicy}
        agent2policy: Dict[str, str]  # {agent_name: policy_name}
    ) -> None:
        """
        Args:
            policy_dict (Dict[str, RLPolicy]): Dict that stores all policies.
            agent2policy (Dict[str, str]): Agent name to policy name mapping.
        """
        self._policy_dict = policy_dict
        self._agent2policy = agent2policy

    def set_policy_states(self, policy_state_dict: Dict[str, object]) -> None:
        """
        Set policies' states.

        Args:
            policy_state_dict: A double-deck dict with format: {policy_name: policy_state}.
        """
        for policy_name, policy_state in policy_state_dict.items():
            policy = self._policy_dict[policy_name]
            policy.set_policy_state(policy_state)

    def choose_actions(self, state_by_agent: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Choose action according to the given (observable) states of all agents.

        Args:
            state_by_agent (Dict[str, np.ndarray]): Agent state dict for all agents. Use agent name as the key to fetch
                states from this dict.

        Returns:
            Dict that contains the action for all agents.
        """
        with torch.no_grad():
            ret = self._choose_actions_impl(state_by_agent)
        return ret

    @abstractmethod
    def _choose_actions_impl(self, state_by_agent: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Implementation of `choose_actions`.
        """
        raise NotImplementedError

    @abstractmethod
    def explore(self) -> None:
        """
        Switch all policies to exploring mode.
        """
        raise NotImplementedError

    @abstractmethod
    def exploit(self) -> None:
        """
        Switch all policies to exploiting mode.
        """
        raise NotImplementedError


class SimpleAgentWrapper(AbsAgentWrapper):
    def __init__(
        self,
        policy_dict: Dict[str, RLPolicy],  # {policy_name: RLPolicy}
        agent2policy: Dict[str, str]  # {agent_name: policy_name}
    ) -> None:
        super(SimpleAgentWrapper, self).__init__(policy_dict=policy_dict, agent2policy=agent2policy)

    def _choose_actions_impl(self, state_by_agent: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Aggregate states by policy
        states_by_policy = defaultdict(list)  # {str: list of np.ndarray}
        agents_by_policy = defaultdict(list)  # {str: list of str}
        for agent_name, state in state_by_agent.items():
            policy_name = self._agent2policy[agent_name]
            states_by_policy[policy_name].append(state)
            agents_by_policy[policy_name].append(agent_name)

        action_dict = {}
        for policy_name in agents_by_policy:
            policy = self._policy_dict[policy_name]
            states = np.vstack(states_by_policy[policy_name])  # np.ndarray
            policy.eval()
            action_dict.update(zip(
                agents_by_policy[policy_name],  # list of str (agent name)
                policy.get_actions(states)  # list of action
            ))
        return action_dict

    def explore(self) -> None:
        for policy in self._policy_dict.values():
            policy.explore()

    def exploit(self) -> None:
        for policy in self._policy_dict.values():
            policy.exploit()


@dataclass
class CacheElement:
    """
    The data structure used to store a cached value during experience collection.
    """
    tick: int
    state: np.ndarray
    agent_state_dict: Dict[str, np.ndarray]
    action_dict: Dict[str, np.ndarray]
    env_action_dict: Dict[str, object]


@dataclass  # TODO: check if all fields are needed
class ExpElement(CacheElement):
    """
    Stores the complete information for a tick. ExpElement is an extension of CacheElement.
    """
    reward_dict: Dict[str, float]
    terminal_dict: Dict[str, bool]
    next_state: np.ndarray = None
    next_agent_state_dict: Dict[str, np.ndarray] = None


# TODO: event typehint
class AbsEnvSampler(object, metaclass=ABCMeta):
    """
    Simulation data collector and policy evaluator.
    """
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
        """
        Args:
            get_env_func (Dict[str, Callable[[str], RLPolicy]]): Dict of functions used to create the learning Env.
            get_policy_func_dict (Dict[str, Callable[[str], RLPolicy]]): Dict of functions used to create policies.
            agent2policy (Dict[str, str]): Agent name to policy name mapping.
            agent_wrapper_cls (Type[AbsAgentWrapper]): Concrete AbsAgentWrapper type.
            reward_eval_delay (int): Number of ticks required after a decision event to evaluate the reward
                for the action taken for that event. Defaults to 0, which means rewards are evaluated immediately
                after executing an action.
            get_test_env_func (Dict[str, Callable[[str], RLPolicy]]): Dict of functions used to create the testing Env.
                If it is None, use the learning Env as the testing Env.
        """
        self._learn_env = get_env_func()
        self._test_env = get_test_env_func() if get_test_env_func is not None else self._learn_env
        self._env: Optional[Env] = None

        self._policy_dict: Dict[str, RLPolicy] = {
            policy_name: func(policy_name) for policy_name, func in get_policy_func_dict.items()
        }
        self._agent_wrapper = agent_wrapper_cls(self._policy_dict, agent2policy)
        self._agent2policy = agent2policy

        # Global state & agent state
        self._state: Optional[np.ndarray] = None
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
    def _translate_to_env_action(self, action_dict: Dict[str, np.ndarray], event) -> Dict[str, object]:
        """
        Translation the actions into the format that the env could recognize.

        Args:
            action_dict (Dict[str, np.ndarray]): Action for all agents.
            event: Decision event.

        Returns:
            A dict that contains env actions for all agents.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, env_action_dict: Dict[str, object], tick: int) -> Dict[str, float]:
        """
        Get rewards according to the env actions.

        Args:
            env_action_dict (Dict[str, object]): Dict that contains env actions for all agents.
            tick (int): Current tick.

        Returns:
            A dict that contains rewards for all agents.
        """
        raise NotImplementedError

    def sample(  # TODO: check logic with qiuyang
        self,
        policy_state_dict: Dict[str, object] = None,
        num_steps: int = -1
    ) -> dict:
        """
        Sample experiences.

        Args:
            policy_state_dict (Dict[str, object]): Policy states dict. If it is not None, then we need to update all
                policies according to the latest policy states, then start the experience collection.
            num_steps (int): Number of collecting steps. Defaults to -1, which means unlimited number of steps.

        Returns:
            A dict that contains the collected experiences and other additional information.
        """
        # Init the env
        self._env = self._learn_env
        if not self._agent_state_dict:
            self._env.reset()
            self._trans_cache.clear()
            _, event, _ = self._env.step(None)
            self._state, self._agent_state_dict = self._get_global_and_agent_state(event)
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
            action_dict = self._agent_wrapper.choose_actions(self._agent_state_dict)
            env_action_dict = self._translate_to_env_action(action_dict, event)
            # Store experiences in the cache
            self._trans_cache.append(
                CacheElement(
                    tick=self._env.tick,
                    state=self._state,
                    agent_state_dict=dict(self._agent_state_dict),
                    action_dict=action_dict,
                    env_action_dict=env_action_dict
                )
            )
            # Update env and get new states (global & agent)
            _, event, done = self._env.step(list(env_action_dict.values()))
            self._state, self._agent_state_dict = (None, {}) if done \
                else self._get_global_and_agent_state(event)
            steps_to_go -= 1

        #
        tick_bound = self._env.tick - self._reward_eval_delay
        experiences = []
        while len(self._trans_cache) > 0 and self._trans_cache[0].tick <= tick_bound:
            cache_element = self._trans_cache.popleft()

            reward_dict = self._get_reward(cache_element.env_action_dict, cache_element.tick)
            self._post_step(cache_element, reward_dict)

            if len(self._trans_cache) > 0:
                next_state = self._trans_cache[0].state
                next_agent_state_dict = dict(self._trans_cache[0].agent_state_dict)
            else:
                next_state = self._state
                next_agent_state_dict = dict(self._agent_state_dict)

            experiences.append(ExpElement(
                tick=cache_element.tick,
                state=cache_element.state,
                agent_state_dict=cache_element.agent_state_dict,
                action_dict=cache_element.action_dict,
                env_action_dict=cache_element.env_action_dict,
                reward_dict=reward_dict,
                terminal_dict={},  # Will be processed later
                next_state=next_state,
                next_agent_state_dict=next_agent_state_dict
            ))

        experiences = self._post_polish_experiences(experiences)

        return {
            "end_of_episode": not self._agent_state_dict,
            "experiences": experiences,
            "tracker": self._tracker
        }

    def _post_polish_experiences(self, experiences: List[ExpElement]) -> List[ExpElement]:
        # Update next_agent_state_dict & terminal_dict by using the entire experience list
        # TODO: Add detailed explanation for this logic block.
        latest_agent_state_dict = {}  # Used to update next_agent_state_dict
        have_log = set([])  # Used to update terminal_dict
        for i in range(len(experiences))[::-1]:
            # Update terminal_dict
            for agent_name in experiences[i].agent_state_dict:
                experiences[i].terminal_dict[agent_name] = (not self._agent_state_dict and agent_name not in have_log)
                have_log.add(agent_name)
            # Update next_agent_state_dict
            for key, value in latest_agent_state_dict.items():
                if key not in experiences[i].next_agent_state_dict:
                    experiences[i].next_agent_state_dict[key] = value
            latest_agent_state_dict.update(experiences[i].next_agent_state_dict)
        return experiences

    def set_policy_states(self, policy_state_dict: Dict[str, object]) -> None:
        self._agent_wrapper.set_policy_states(policy_state_dict)

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
            action_dict = self._agent_wrapper.choose_actions(agent_state_dict)
            env_action_dict = self._translate_to_env_action(action_dict, event)
            _, event, terminal = self._env.step(list(env_action_dict.values()))
            if not terminal:
                _, agent_state_dict = self._get_global_and_agent_state(event)
        return self._tracker

    @abstractmethod
    def _post_step(
        self, cache_element: CacheElement, reward: Dict[str, float]
    ) -> None:
        raise NotImplementedError
