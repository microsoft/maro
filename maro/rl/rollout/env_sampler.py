# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import collections
import os
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch

from maro.rl.policy import AbsPolicy, RLPolicy
from maro.rl.utils.objects import FILE_SUFFIX
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

    def set_policy_state(self, policy_state_dict: Dict[str, dict]) -> None:
        """Set policies' states.

        Args:
            policy_state_dict (Dict[str, dict]): Double-deck dict with format: {policy_name: policy_state}.
        """
        for policy_name, policy_state in policy_state_dict.items():
            policy = self._policy_dict[policy_name]
            if isinstance(policy, RLPolicy):
                policy.set_state(policy_state)

    def choose_actions(
        self,
        state_by_agent: Dict[Any, Union[np.ndarray, list]],
    ) -> Dict[Any, Union[np.ndarray, list]]:
        """Choose action according to the given (observable) states of all agents.

        Args:
            state_by_agent (Dict[Any, Union[np.ndarray, list]]): Dictionary containing each agent's states.
                If the policy is a `RLPolicy`, its state is a Numpy array. Otherwise, its state is a list of objects.

        Returns:
            actions (Dict[Any, Union[np.ndarray, list]]): Dict that contains the action for all agents.
                If the policy is a `RLPolicy`, its action is a Numpy array. Otherwise, its action is a list of objects.
        """
        self.switch_to_eval_mode()
        with torch.no_grad():
            ret = self._choose_actions_impl(state_by_agent)
        return ret

    @abstractmethod
    def _choose_actions_impl(
        self,
        state_by_agent: Dict[Any, Union[np.ndarray, list]],
    ) -> Dict[Any, Union[np.ndarray, list]]:
        """Implementation of `choose_actions`."""
        raise NotImplementedError

    @abstractmethod
    def explore(self) -> None:
        """Switch all policies to exploration mode."""
        raise NotImplementedError

    @abstractmethod
    def exploit(self) -> None:
        """Switch all policies to exploitation mode."""
        raise NotImplementedError

    @abstractmethod
    def switch_to_eval_mode(self) -> None:
        """Switch the environment sampler to evaluation mode."""
        raise NotImplementedError


class SimpleAgentWrapper(AbsAgentWrapper):
    def __init__(
        self,
        policy_dict: Dict[str, AbsPolicy],  # {policy_name: AbsPolicy}
        agent2policy: Dict[Any, str],  # {agent_name: policy_name}
    ) -> None:
        super(SimpleAgentWrapper, self).__init__(policy_dict=policy_dict, agent2policy=agent2policy)

    def _choose_actions_impl(
        self,
        state_by_agent: Dict[Any, Union[np.ndarray, list]],
    ) -> Dict[Any, Union[np.ndarray, list]]:
        # Aggregate states by policy
        states_by_policy = collections.defaultdict(list)  # {str: list of np.ndarray}
        agents_by_policy = collections.defaultdict(list)  # {str: list of str}
        for agent_name, state in state_by_agent.items():
            policy_name = self._agent2policy[agent_name]
            states_by_policy[policy_name].append(state)
            agents_by_policy[policy_name].append(agent_name)

        action_dict: dict = {}
        for policy_name in agents_by_policy:
            policy = self._policy_dict[policy_name]

            if isinstance(policy, RLPolicy):
                states = np.vstack(states_by_policy[policy_name])  # np.ndarray
            else:
                states = states_by_policy[policy_name]  # list
            actions: Union[np.ndarray, list] = policy.get_actions(states)  # np.ndarray or list
            action_dict.update(zip(agents_by_policy[policy_name], actions))

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
class ExpElement:
    """Stores the complete information for a tick."""

    tick: int
    state: np.ndarray
    agent_state_dict: Dict[Any, np.ndarray]
    action_dict: Dict[Any, np.ndarray]
    reward_dict: Dict[Any, float]
    terminal_dict: Dict[Any, bool]
    next_state: Optional[np.ndarray]
    next_agent_state_dict: Dict[Any, np.ndarray]

    @property
    def agent_names(self) -> list:
        return sorted(self.agent_state_dict.keys())

    @property
    def num_agents(self) -> int:
        return len(self.agent_state_dict)

    def split_contents_by_agent(self) -> Dict[Any, ExpElement]:
        ret = {}
        for agent_name in self.agent_state_dict.keys():
            ret[agent_name] = ExpElement(
                tick=self.tick,
                state=self.state,
                agent_state_dict={agent_name: self.agent_state_dict[agent_name]},
                action_dict={agent_name: self.action_dict[agent_name]},
                reward_dict={agent_name: self.reward_dict[agent_name]},
                terminal_dict={agent_name: self.terminal_dict[agent_name]},
                next_state=self.next_state,
                next_agent_state_dict={
                    agent_name: self.next_agent_state_dict[agent_name],
                }
                if self.next_agent_state_dict is not None and agent_name in self.next_agent_state_dict
                else {},
            )
        return ret

    def split_contents_by_trainer(self, agent2trainer: Dict[Any, str]) -> Dict[str, ExpElement]:
        """Split the ExpElement's contents by trainer.

        Args:
            agent2trainer (Dict[Any, str]): Mapping of agent name and trainer name.

        Returns:
            Contents (Dict[str, ExpElement]): A dict that contains the ExpElements of all trainers. The key of this
                dict is the trainer name.
        """
        ret: Dict[str, ExpElement] = collections.defaultdict(
            lambda: ExpElement(
                tick=self.tick,
                state=self.state,
                agent_state_dict={},
                action_dict={},
                reward_dict={},
                terminal_dict={},
                next_state=self.next_state,
                next_agent_state_dict=None if self.next_agent_state_dict is None else {},
            ),
        )
        for agent_name, trainer_name in agent2trainer.items():
            if agent_name in self.agent_state_dict:
                ret[trainer_name].agent_state_dict[agent_name] = self.agent_state_dict[agent_name]
                ret[trainer_name].action_dict[agent_name] = self.action_dict[agent_name]
                ret[trainer_name].reward_dict[agent_name] = self.reward_dict[agent_name]
                ret[trainer_name].terminal_dict[agent_name] = self.terminal_dict[agent_name]
                if self.next_agent_state_dict is not None and agent_name in self.next_agent_state_dict:
                    ret[trainer_name].next_agent_state_dict[agent_name] = self.next_agent_state_dict[agent_name]
        return ret


@dataclass
class CacheElement(ExpElement):
    event: Any
    env_action_dict: Dict[Any, np.ndarray]

    def make_exp_element(self) -> ExpElement:
        assert len(self.terminal_dict) == len(self.agent_state_dict) == len(self.action_dict)
        assert len(self.terminal_dict) == len(self.next_agent_state_dict) == len(self.reward_dict)

        return ExpElement(
            tick=self.tick,
            state=self.state,
            agent_state_dict=self.agent_state_dict,
            action_dict=self.action_dict,
            reward_dict=self.reward_dict,
            terminal_dict=self.terminal_dict,
            next_state=self.next_state,
            next_agent_state_dict=self.next_agent_state_dict,
        )


class AbsEnvSampler(object, metaclass=ABCMeta):
    """Simulation data collector and policy evaluator.

    Args:
        learn_env (Env): Environment used for training.
        test_env (Env): Environment used for testing.
        policies (List[AbsPolicy]): List of policies.
        agent2policy (Dict[Any, str]): Agent name to policy name mapping of the RL job.
        trainable_policies (List[str]): Name of trainable policies.
        agent_wrapper_cls (Type[AbsAgentWrapper], default=SimpleAgentWrapper): Specific AgentWrapper type.
        reward_eval_delay (int, default=None): Number of ticks required after a decision event to evaluate the reward
            for the action taken for that event. If it is None, calculate reward immediately after `step()`.
    """

    def __init__(
        self,
        learn_env: Env,
        test_env: Env,
        policies: List[AbsPolicy],
        agent2policy: Dict[Any, str],
        trainable_policies: List[str] = None,
        agent_wrapper_cls: Type[AbsAgentWrapper] = SimpleAgentWrapper,
        reward_eval_delay: int = None,
    ) -> None:
        self._learn_env = learn_env
        self._test_env = test_env

        self._agent_wrapper_cls = agent_wrapper_cls

        self._event: Optional[list] = None
        self._end_of_episode = True
        self._state: Optional[np.ndarray] = None
        self._agent_state_dict: Dict[Any, np.ndarray] = {}

        self._trans_cache: List[CacheElement] = []
        self._agent_last_index: Dict[Any, int] = {}  # Index of last occurrence of agent in self._trans_cache
        self._reward_eval_delay = reward_eval_delay

        self._info: dict = {}

        assert self._reward_eval_delay is None or self._reward_eval_delay >= 0

        #
        self._env: Optional[Env] = None
        self._policy_dict: Dict[str, AbsPolicy] = {policy.name: policy for policy in policies}
        self._rl_policy_dict: Dict[str, RLPolicy] = {
            policy.name: policy for policy in policies if isinstance(policy, RLPolicy)
        }
        self._agent2policy = agent2policy
        self._agent_wrapper = self._agent_wrapper_cls(self._policy_dict, self._agent2policy)

        if trainable_policies is not None:
            self._trainable_policies = trainable_policies
        else:
            self._trainable_policies = list(self._policy_dict.keys())  # Default: all policies are trainable
        self._trainable_agents = {
            agent_id for agent_id, policy_name in self._agent2policy.items() if policy_name in self._trainable_policies
        }

        assert all(
            [policy_name in self._rl_policy_dict for policy_name in self._trainable_policies],
        ), "All trainable policies must be RL policies!"

    @property
    def env(self) -> Env:
        assert self._env is not None
        return self._env

    def _switch_env(self, env: Env) -> None:
        self._env = env

    def assign_policy_to_device(self, policy_name: str, device: torch.device) -> None:
        self._rl_policy_dict[policy_name].to_device(device)

    def _get_global_and_agent_state(
        self,
        event: Any,
        tick: int = None,
    ) -> Tuple[Optional[Any], Dict[Any, Union[np.ndarray, list]]]:
        """Get the global and individual agents' states.

        Args:
            event (Any): Event.
            tick (int, default=None): Current tick.

        Returns:
            Global state (Optional[Any])
            Dict of agent states (Dict[Any, Union[np.ndarray, list]]). If the policy is a `RLPolicy`,
                its state is a Numpy array. Otherwise, its state is a list of objects.
        """
        global_state, agent_state_dict = self._get_global_and_agent_state_impl(event, tick)
        for agent_name, state in agent_state_dict.items():
            policy_name = self._agent2policy[agent_name]
            policy = self._policy_dict[policy_name]
            if isinstance(policy, RLPolicy) and not isinstance(state, np.ndarray):
                raise ValueError(f"Agent {agent_name} uses a RLPolicy but its state is not a np.ndarray.")
        return global_state, agent_state_dict

    @abstractmethod
    def _get_global_and_agent_state_impl(
        self,
        event: Any,
        tick: int = None,
    ) -> Tuple[Union[None, np.ndarray, list], Dict[Any, Union[np.ndarray, list]]]:
        raise NotImplementedError

    @abstractmethod
    def _translate_to_env_action(
        self,
        action_dict: Dict[Any, Union[np.ndarray, list]],
        event: Any,
    ) -> dict:
        """Translate model-generated actions into an object that can be executed by the env.

        Args:
            action_dict (Dict[Any, Union[np.ndarray, list]]): Action for all agents. If the policy is a
                `RLPolicy`, its (input) action is a Numpy array. Otherwise, its (input) action is a list of objects.
            event (Any): Decision event.

        Returns:
            A dict that contains env actions for all agents.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, env_action_dict: dict, event: Any, tick: int) -> Dict[Any, float]:
        """Get rewards according to the env actions.

        Args:
            env_action_dict (dict): Dict that contains env actions for all agents.
            event (Any): Decision event.
            tick (int): Current tick.

        Returns:
            A dict that contains rewards for all agents.
        """
        raise NotImplementedError

    def _step(self, actions: Optional[list]) -> None:
        _, self._event, self._end_of_episode = self.env.step(actions)
        self._state, self._agent_state_dict = (
            (None, {}) if self._end_of_episode else self._get_global_and_agent_state(self._event)
        )

    def _calc_reward(self, cache_element: CacheElement) -> None:
        cache_element.reward_dict = self._get_reward(
            cache_element.env_action_dict,
            cache_element.event,
            cache_element.tick,
        )
        cache_element.reward_dict = {agent: cache_element.reward_dict[agent] for agent in cache_element.agent_names}

    def _append_cache_element(self, cache_element: Optional[CacheElement]) -> None:
        """`cache_element` == None means we are processing the last element in trans_cache"""
        if cache_element is None:
            if len(self._trans_cache) > 0:
                self._trans_cache[-1].next_state = self._trans_cache[-1].state

            for agent_name, i in self._agent_last_index.items():
                e = self._trans_cache[i]
                e.terminal_dict[agent_name] = self._end_of_episode
                e.next_agent_state_dict[agent_name] = e.agent_state_dict[agent_name]
        else:
            self._trans_cache.append(cache_element)

            if len(self._trans_cache) > 0:
                self._trans_cache[-1].next_state = cache_element.state

            cur_index = len(self._trans_cache) - 1
            for agent_name in cache_element.agent_names:
                if agent_name in self._agent_last_index:
                    i = self._agent_last_index[agent_name]
                    self._trans_cache[i].terminal_dict[agent_name] = False
                    self._trans_cache[i].next_agent_state_dict[agent_name] = cache_element.agent_state_dict[agent_name]
                self._agent_last_index[agent_name] = cur_index

    def _reset(self) -> None:
        self.env.reset()
        self._info.clear()
        self._trans_cache.clear()
        self._agent_last_index.clear()
        self._step(None)

    def _select_trainable_agents(self, original_dict: dict) -> dict:
        return {k: v for k, v in original_dict.items() if k in self._trainable_agents}

    def sample(
        self,
        policy_state: Optional[Dict[str, Dict[str, Any]]] = None,
        num_steps: Optional[int] = None,
    ) -> dict:
        """Sample experiences.

        Args:
            policy_state (Dict[str, dict]): Policy state dict. If it is not None, then we need to update all
                policies according to the latest policy states, then start the experience collection.
            num_steps (Optional[int], default=None): Number of collecting steps. If it is None, interactions with
                the environment will continue until the terminal state is reached.

        Returns:
            A dict that contains the collected experiences and additional information.
        """
        # Init the env
        self._switch_env(self._learn_env)
        if self._end_of_episode:
            self._reset()

        # Update policy state if necessary
        if policy_state is not None:
            self.set_policy_state(policy_state)

        # Collect experience
        self._agent_wrapper.explore()
        steps_to_go = float("inf") if num_steps is None else num_steps
        while not self._end_of_episode and steps_to_go > 0:
            # Get agent actions and translate them to env actions
            action_dict = self._agent_wrapper.choose_actions(self._agent_state_dict)
            env_action_dict = self._translate_to_env_action(action_dict, self._event)

            # Store experiences in the cache
            cache_element = CacheElement(
                tick=self.env.tick,
                event=self._event,
                state=self._state,
                agent_state_dict=self._select_trainable_agents(self._agent_state_dict),
                action_dict=self._select_trainable_agents(action_dict),
                env_action_dict=self._select_trainable_agents(env_action_dict),
                # The following will be generated later
                reward_dict={},
                terminal_dict={},
                next_state=None,
                next_agent_state_dict={},
            )

            # Update env and get new states (global & agent)
            self._step(list(env_action_dict.values()))

            if self._reward_eval_delay is None:
                self._calc_reward(cache_element)
                self._post_step(cache_element)
            self._append_cache_element(cache_element)
            steps_to_go -= 1
        self._append_cache_element(None)

        tick_bound = self.env.tick - (0 if self._reward_eval_delay is None else self._reward_eval_delay)
        experiences: List[ExpElement] = []
        while len(self._trans_cache) > 0 and self._trans_cache[0].tick <= tick_bound:
            cache_element = self._trans_cache.pop(0)
            # !: Here the reward calculation method requires the given tick is enough and must be used then.
            if self._reward_eval_delay is not None:
                self._calc_reward(cache_element)
                self._post_step(cache_element)
            experiences.append(cache_element.make_exp_element())

        self._agent_last_index = {
            k: v - len(experiences) for k, v in self._agent_last_index.items() if v >= len(experiences)
        }

        return {
            "end_of_episode": self._end_of_episode,
            "experiences": [experiences],
            "info": [deepcopy(self._info)],  # TODO: may have overhead issues. Leave to future work.
        }

    def set_policy_state(self, policy_state_dict: Dict[str, dict]) -> None:
        """Set policies' states.

        Args:
            policy_state_dict (Dict[str, dict]): Double-deck dict with format: {policy_name: policy_state}.
        """
        self._agent_wrapper.set_policy_state(policy_state_dict)

    def load_policy_state(self, path: str) -> List[str]:
        file_list = os.listdir(path)
        policy_state_dict = {}
        loaded = []
        for file_name in file_list:
            if "non_policy" in file_name or not file_name.endswith(f"_policy.{FILE_SUFFIX}"):  # TODO: remove hardcode
                continue
            policy_name, policy_state = torch.load(os.path.join(path, file_name))
            policy_state_dict[policy_name] = policy_state
            loaded.append(policy_name)
        self.set_policy_state(policy_state_dict)

        return loaded

    def eval(self, policy_state: Dict[str, Dict[str, Any]] = None) -> dict:
        self._switch_env(self._test_env)
        self._reset()
        if policy_state is not None:
            self.set_policy_state(policy_state)

        self._agent_wrapper.exploit()
        while not self._end_of_episode:
            action_dict = self._agent_wrapper.choose_actions(self._agent_state_dict)
            env_action_dict = self._translate_to_env_action(action_dict, self._event)

            # Store experiences in the cache
            cache_element = CacheElement(
                tick=self.env.tick,
                event=self._event,
                state=self._state,
                agent_state_dict=self._select_trainable_agents(self._agent_state_dict),
                action_dict=self._select_trainable_agents(action_dict),
                env_action_dict=self._select_trainable_agents(env_action_dict),
                # The following will be generated later
                reward_dict={},
                terminal_dict={},
                next_state=None,
                next_agent_state_dict={},
            )

            # Update env and get new states (global & agent)
            self._step(list(env_action_dict.values()))

            if self._reward_eval_delay is None:  # TODO: necessary to calculate reward in eval()?
                self._calc_reward(cache_element)
                self._post_eval_step(cache_element)

            self._append_cache_element(cache_element)
        self._append_cache_element(None)

        tick_bound = self.env.tick - (0 if self._reward_eval_delay is None else self._reward_eval_delay)
        while len(self._trans_cache) > 0 and self._trans_cache[0].tick <= tick_bound:
            cache_element = self._trans_cache.pop(0)
            if self._reward_eval_delay is not None:
                self._calc_reward(cache_element)
                self._post_eval_step(cache_element)

        return {"info": [self._info]}

    @abstractmethod
    def _post_step(self, cache_element: CacheElement) -> None:
        raise NotImplementedError

    @abstractmethod
    def _post_eval_step(self, cache_element: CacheElement) -> None:
        raise NotImplementedError

    def post_collect(self, info_list: list, ep: int) -> None:
        """Routines to be invoked at the end of training episodes"""

    def post_evaluate(self, info_list: list, ep: int) -> None:
        """Routines to be invoked at the end of evaluation episodes"""
