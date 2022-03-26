# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import collections
from abc import ABCMeta, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from maro.rl.policy import AbsPolicy, RLPolicy
from maro.simulator import Env


class AbsAgentWrapper(object, metaclass=ABCMeta):
    """Agent wrapper. Used to manager agents & policies during experience collection.

    Args:
        policy_dict (Dict[str, AbsPolicy]): Dictionary that maps policy names to policy instances.
        agent2policy (Dict[Any, str]): Agent name to policy name mapping.
    """

    def __init__(
        self,
        policy_dict: Dict[str, AbsPolicy],  # {policy_name: AbsPolicy}
        agent2policy: Dict[Any, str],  # {agent_name: policy_name}
    ) -> None:
        self._policy_dict = policy_dict
        self._agent2policy = agent2policy

    def set_policy_state(self, policy_state_dict: Dict[str, object]) -> None:
        """Set policies' states.

        Args:
            policy_state_dict (Dict[str, object]): Double-deck dict with format: {policy_name: policy_state}.
        """
        for policy_name, policy_state in policy_state_dict.items():
            policy = self._policy_dict[policy_name]
            if isinstance(policy, RLPolicy):
                policy.set_state(policy_state)

    def choose_actions(self, state_by_agent: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        """Choose action according to the given (observable) states of all agents.

        Args:
            state_by_agent (Dict[Any, np.ndarray]): Dictionary containing each agent's state vector.
                The keys are agent names.

        Returns:
            actions (Dict[Any, np.ndarray]): Dict that contains the action for all agents.
        """
        self.switch_to_eval_mode()
        with torch.no_grad():
            ret = self._choose_actions_impl(state_by_agent)
        return ret

    @abstractmethod
    def _choose_actions_impl(self, state_by_agent: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        """Implementation of `choose_actions`.
        """
        raise NotImplementedError

    @abstractmethod
    def explore(self) -> None:
        """Switch all policies to exploration mode.
        """
        raise NotImplementedError

    @abstractmethod
    def exploit(self) -> None:
        """Switch all policies to exploitation mode.
        """
        raise NotImplementedError

    @abstractmethod
    def switch_to_eval_mode(self) -> None:
        """Switch the environment sampler to evaluation mode.
        """
        pass


class SimpleAgentWrapper(AbsAgentWrapper):
    def __init__(
        self,
        policy_dict: Dict[str, RLPolicy],  # {policy_name: RLPolicy}
        agent2policy: Dict[Any, str],  # {agent_name: policy_name}
    ) -> None:
        super(SimpleAgentWrapper, self).__init__(policy_dict=policy_dict, agent2policy=agent2policy)

    def _choose_actions_impl(self, state_by_agent: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
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
            action_dict.update(zip(
                agents_by_policy[policy_name],  # list of str (agent name)
                policy.get_actions(states),  # list of action
            ))
        return action_dict

    def explore(self) -> None:
        for policy in self._policy_dict.values():
            policy.explore()

    def exploit(self) -> None:
        for policy in self._policy_dict.values():
            policy.exploit()

    def switch_to_eval_mode(self) -> None:
        for policy in self._policy_dict.values():
            policy.eval()


@dataclass
class CacheElement:
    """Raw transition information that can be post-processed into an `ExpElement`.
    """
    tick: int
    event: object
    state: np.ndarray
    agent_state_dict: Dict[Any, np.ndarray]
    action_dict: Dict[Any, np.ndarray]
    env_action_dict: Dict[Any, object]


@dataclass
class ExpElement:
    """Stores the complete information for a tick.
    """
    tick: int
    state: np.ndarray
    agent_state_dict: Dict[Any, np.ndarray]
    action_dict: Dict[Any, np.ndarray]
    reward_dict: Dict[Any, float]
    terminal_dict: Dict[Any, bool]
    next_state: Optional[np.ndarray]
    next_agent_state_dict: Optional[Dict[Any, np.ndarray]]

    @property
    def agent_names(self) -> list:
        return sorted(self.agent_state_dict.keys())

    @property
    def num_agents(self) -> int:
        return len(self.agent_state_dict)

    def split_contents(self, agent2trainer: Dict[Any, str]) -> Dict[str, ExpElement]:
        """Split the ExpElement's contents by trainer.

        Args:
            agent2trainer (Dict[Any, str]): Mapping of agent name and trainer name.

        Returns:
            Contents (Dict[str, ExpElement]): A dict that contains the ExpElements of all trainers. The key of this
                dict is the trainer name.
        """
        ret = collections.defaultdict(lambda: ExpElement(
            tick=self.tick,
            state=self.state,
            agent_state_dict={},
            action_dict={},
            reward_dict={},
            terminal_dict={},
            next_state=self.next_state,
            next_agent_state_dict=None if self.next_agent_state_dict is None else {},
        ))
        for agent_name, trainer_name in agent2trainer.items():
            if agent_name in self.agent_state_dict:
                ret[trainer_name].agent_state_dict[agent_name] = self.agent_state_dict[agent_name]
                ret[trainer_name].action_dict[agent_name] = self.action_dict[agent_name]
                ret[trainer_name].reward_dict[agent_name] = self.reward_dict[agent_name]
                ret[trainer_name].terminal_dict[agent_name] = self.terminal_dict[agent_name]
                if self.next_agent_state_dict is not None and agent_name in self.next_agent_state_dict:
                    ret[trainer_name].next_agent_state_dict[agent_name] = self.next_agent_state_dict[agent_name]
        return ret


class AbsEnvSampler(object, metaclass=ABCMeta):
    """Simulation data collector and policy evaluator.

    Args:
        get_env (Callable[[], Env]): Function used to create the rollout environment.
        policy_creator (Dict[str, Callable[[str], AbsPolicy]]): Dict of functions to create policies by name.
        agent2policy (Dict[Any, str]): Mapping of agent name to policy name.
        trainable_policies (List[str], default=None): List of trainable policy names. Experiences generated using the
            policies specified in this list will be collected and passed to a training manager for training. Defaults
            to None, in which case all policies are trainable.
        agent_wrapper_cls (Type[AbsAgentWrapper], default=SimpleAgentWrapper): Specific AgentWrapper type.
        reward_eval_delay (int): Number of ticks required after a decision event to evaluate the reward
            for the action taken for that event.
        get_test_env (Callable[[], Env], default=None): Function used to create the testing environment. If it is None,
            reuse the rollout environment as the testing environment.
    """

    def __init__(
        self,
        get_env: Callable[[], Env],
        policy_creator: Dict[str, Callable[[str], AbsPolicy]],
        agent2policy: Dict[Any, str],  # {agent_name: policy_name}
        trainable_policies: List[str] = None,
        agent_wrapper_cls: Type[AbsAgentWrapper] = SimpleAgentWrapper,
        reward_eval_delay: int = 0,
        get_test_env: Callable[[], Env] = None,
    ) -> None:
        self._learn_env = get_env()
        self._test_env = get_test_env() if get_test_env is not None else self._learn_env
        self._env: Optional[Env] = None
        self._event = None  # Need this to remember the last event if an episode is divided into multiple segments

        self._policy_dict: Dict[str, AbsPolicy] = {
            policy_name: func(policy_name) for policy_name, func in policy_creator.items()
        }
        self._rl_policy_dict: Dict[str, AbsPolicy] = {
            name: policy for name, policy in self._policy_dict.items() if isinstance(policy, RLPolicy)
        }
        self._agent2policy = agent2policy
        self._agent_wrapper = agent_wrapper_cls(self._policy_dict, agent2policy)
        if trainable_policies is None:
            self._trainable_policies = set(self._policy_dict.keys())
        else:
            self._trainable_policies = set(trainable_policies)
        self._trainable_agents = {
            agent_id for agent_id, policy_name in self._agent2policy.items() if policy_name in self._trainable_policies
        }

        # Global state & agent state
        self._state: Optional[np.ndarray] = None
        self._agent_state_dict: Dict[Any, np.ndarray] = {}

        self._trans_cache: Deque[CacheElement] = deque()
        self._reward_eval_delay = reward_eval_delay

        self._info = {}

    @property
    def rl_policy_dict(self) -> Dict[str, RLPolicy]:
        return self._rl_policy_dict

    @abstractmethod
    def _get_global_and_agent_state(
        self, event: object, tick: int = None,
    ) -> Tuple[Optional[np.ndarray], Dict[Any, np.ndarray]]:
        """Get the global and individual agents' states.

        Args:
            event (object): Event.
            tick (int, default=None): Current tick.

        Returns:
            Global state (np.ndarray)
            Dict of agent states (Dict[Any, np.ndarray])
        """
        raise NotImplementedError

    @abstractmethod
    def _translate_to_env_action(self, action_dict: Dict[Any, np.ndarray], event: object) -> Dict[Any, object]:
        """Translate model-generated actions into an object that can be executed by the env.

        Args:
            action_dict (Dict[Any, np.ndarray]): Action for all agents.
            event (object): Decision event.

        Returns:
            A dict that contains env actions for all agents.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, env_action_dict: Dict[Any, object], event: object, tick: int) -> Dict[Any, float]:
        """Get rewards according to the env actions.

        Args:
            env_action_dict (Dict[Any, object]): Dict that contains env actions for all agents.
            event (object): Decision event.
            tick (int): Current tick.

        Returns:
            A dict that contains rewards for all agents.
        """
        raise NotImplementedError

    def sample(self, policy_state: Optional[Dict[str, object]] = None, num_steps: Optional[int] = None) -> dict:
        """Sample experiences.

        Args:
            policy_state (Dict[str, object]): Policy state dict. If it is not None, then we need to update all
                policies according to the latest policy states, then start the experience collection.
            num_steps (Optional[int], default=None): Number of collecting steps. If it is None, interactions with
                the environment will continue until the terminal state is reached.

        Returns:
            A dict that contains the collected experiences and additional information.
        """
        # Init the env
        self._env = self._learn_env
        if not self._agent_state_dict:
            self._env.reset()
            self._info.clear()
            self._trans_cache.clear()
            _, self._event, _ = self._env.step(None)
            self._state, self._agent_state_dict = self._get_global_and_agent_state(self._event)

        # Update policy state if necessary
        if policy_state is not None:
            self.set_policy_state(policy_state)

        # Collect experience
        self._agent_wrapper.explore()
        steps_to_go = float("inf") if num_steps is None else num_steps
        while self._agent_state_dict and steps_to_go > 0:
            # Get agent actions and translate them to env actions
            action_dict = self._agent_wrapper.choose_actions(self._agent_state_dict)
            env_action_dict = self._translate_to_env_action(action_dict, self._event)

            # Store experiences in the cache
            self._trans_cache.append(
                CacheElement(
                    tick=self._env.tick,
                    event=self._event,
                    state=self._state,
                    agent_state_dict={
                        id_: state for id_, state in self._agent_state_dict.items() if id_ in self._trainable_agents
                    },
                    action_dict={id_: action for id_, action in action_dict.items() if id_ in self._trainable_agents},
                    env_action_dict={
                        id_: env_action for id_, env_action in env_action_dict.items() if id_ in self._trainable_agents
                    },
                )
            )

            # Update env and get new states (global & agent)
            _, self._event, is_done = self._env.step(list(env_action_dict.values()))
            self._state, self._agent_state_dict = (None, {}) if is_done \
                else self._get_global_and_agent_state(self._event)
            steps_to_go -= 1

        tick_bound = self._env.tick - self._reward_eval_delay
        experiences = []
        while len(self._trans_cache) > 0 and self._trans_cache[0].tick <= tick_bound:
            cache_element = self._trans_cache.popleft()
            # !: Here the reward calculation method requires the given tick is enough and must be used then.
            reward_dict = self._get_reward(cache_element.env_action_dict, cache_element.event, cache_element.tick)
            self._post_step(cache_element, reward_dict)
            if len(self._trans_cache) > 0:
                next_state = self._trans_cache[0].state
                next_agent_state_dict = dict(self._trans_cache[0].agent_state_dict)
            else:
                next_state = self._state
                next_agent_state_dict = {
                    id_: state for id_, state in self._agent_state_dict.items() if id_ in self._trainable_agents
                }

            experiences.append(ExpElement(
                tick=cache_element.tick,
                state=cache_element.state,
                agent_state_dict=cache_element.agent_state_dict,
                action_dict=cache_element.action_dict,
                reward_dict=reward_dict,
                terminal_dict={},  # Will be processed later in `_post_polish_experiences()`
                next_state=next_state,
                next_agent_state_dict=next_agent_state_dict,
            ))

        experiences = self._post_polish_experiences(experiences)

        return {
            "end_of_episode": not self._agent_state_dict,
            "experiences": [experiences],
            "info": [self._info],
        }

    def _post_polish_experiences(self, experiences: List[ExpElement]) -> List[ExpElement]:
        """Update next_agent_state_dict & terminal_dict using the entire experience list.

        Args:
            experiences (List[ExpElement]): Sequence of ExpElements.

        Returns:
            The update sequence of ExpElements.
        """
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
            latest_agent_state_dict.update(experiences[i].agent_state_dict)
        return experiences

    def set_policy_state(self, policy_state_dict: Dict[str, object]) -> None:
        """Set policies' states.

        Args:
            policy_state_dict (Dict[str, object]): Double-deck dict with format: {policy_name: policy_state}.
        """
        self._agent_wrapper.set_policy_state(policy_state_dict)

    def eval(self, policy_state: Dict[str, object] = None) -> dict:
        self._env = self._test_env
        if policy_state is not None:
            self.set_policy_state(policy_state)

        self._agent_wrapper.exploit()
        self._env.reset()
        is_done = False
        _, self._event, _ = self._env.step(None)
        self._state, self._agent_state_dict = self._get_global_and_agent_state(self._event)
        while not is_done:
            action_dict = self._agent_wrapper.choose_actions(self._agent_state_dict)
            env_action_dict = self._translate_to_env_action(action_dict, self._event)

            # Store experiences in the cache
            self._trans_cache.append(
                CacheElement(
                    tick=self._env.tick,
                    event=self._event,
                    state=self._state,
                    agent_state_dict={
                        id_: state for id_, state in self._agent_state_dict.items() if id_ in self._trainable_agents
                    },
                    action_dict={id_: action for id_, action in action_dict.items() if id_ in self._trainable_agents},
                    env_action_dict={
                        id_: env_action for id_, env_action in env_action_dict.items() if id_ in self._trainable_agents
                    },
                )
            )
            # Update env and get new states (global & agent)
            _, self._event, is_done = self._env.step(list(env_action_dict.values()))
            self._state, self._agent_state_dict = (None, {}) if is_done \
                else self._get_global_and_agent_state(self._event)

        tick_bound = self._env.tick - self._reward_eval_delay
        while self._trans_cache and self._trans_cache[0].tick <= tick_bound:
            cache_element = self._trans_cache.popleft()
            reward_dict = self._get_reward(cache_element.env_action_dict, cache_element.event, cache_element.tick)
            self._post_eval_step(cache_element, reward_dict)

        return {"info": [self._info]}

    @abstractmethod
    def _post_step(self, cache_element: CacheElement, reward: Dict[Any, float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _post_eval_step(self, cache_element: CacheElement, reward: Dict[Any, float]) -> None:
        raise NotImplementedError
