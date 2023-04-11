# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from abc import abstractmethod
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import torch
from tianshou.data import Batch

from maro.rl_v31.policy.base import AbsPolicy, BaseRLPolicy
from maro.simulator import Env
from maro.rl_v31.objects import CacheElement, ExpElement
from .typing import ActionType, EnvStepRes, ObservationType, PolicyActionType, StateType


class AgentWrapper(object):
    def __init__(
        self,
        policy_dict: Dict[str, AbsPolicy],  # {policy_name: AbsPolicy}
        agent2policy: Dict[Any, str],  # {agent_name: policy_name}
    ) -> None:
        self._policy_dict = policy_dict
        self._agent2policy = agent2policy

    def set_policy_state(self, policy_state_dict: Optional[Dict[str, dict]]) -> None:
        if policy_state_dict is not None:
            for name, state_dict in policy_state_dict.items():
                self._policy_dict[name].set_states(state_dict)

    def eval(self) -> None:
        for policy in self._policy_dict.values():
            policy.eval()

    def choose_actions(
        self,
        batch_by_agent: Dict[Any, Batch],
    ) -> Dict[Any, PolicyActionType]:
        self.eval()
        with torch.no_grad():
            policy2batches = defaultdict(deque)
            policy2agents = defaultdict(deque)
            for agent_name, batch in batch_by_agent.items():
                policy_name = self._agent2policy[agent_name]
                policy2batches[policy_name].append(batch)
                policy2agents[policy_name].append(agent_name)

            act_dict = {}
            for policy_name, batches in policy2batches.items():
                policy = self._policy_dict[policy_name]
                sizes = list(map(len, batches))
                agents = policy2agents[policy_name]
                batch = Batch.cat(batches)

                act = policy(batch).act
                i = 0
                for agent_name, size in zip(agents, sizes):
                    act_dict[agent_name] = act[i : i + size]
                    i += size

        return act_dict

    def switch_explore(self, explore: bool) -> None:
        for policy in self._policy_dict.values():
            if isinstance(policy, BaseRLPolicy):
                policy.switch_explore(explore)

    def save(self, path: str) -> None:
        for policy in self._policy_dict.values():
            torch.save(policy.get_states(), os.path.join(path, f"policy__{policy.name}.ckpt"))

    def load(self, path: str) -> None:
        for policy in self._policy_dict.values():
            policy.set_states(torch.load(os.path.join(path, f"policy__{policy.name}.ckpt")))


class EnvWrapper(object):
    def __init__(
        self,
        env: Env,
        reward_eval_delay: Optional[int] = None,
        max_episode_length: Optional[int] = None,  # TODO: move to collector?
        discard_tail_elements: bool = False,
    ) -> None:
        self.env = env
        self._reward_eval_delay = reward_eval_delay or 0
        self._max_episode_length = max_episode_length
        self._discard_tail_elements = discard_tail_elements

        self.cur_ep_length = 0
        self.event: Optional[Any] = None
        self.end_of_episode = True
        self.obs: Optional[ObservationType] = None
        self.agent_obs_dict: Dict[Any, ObservationType] = {}

        self._all_elements: List[CacheElement] = []
        self._agent_last_idx: Dict[Any, int] = {}

        self._total_num_interaction = 0

        # Elements in [0, n_ready_elements) contains reward, so they are ready to be returned
        self.n_ready_elements = 0

    def collect_ready_exps(self) -> List[ExpElement]:
        ready_elements = [e.to_exp_element() for e in self._all_elements[: self.n_ready_elements]]
        self._all_elements = self._all_elements[self.n_ready_elements :]

        self._agent_last_idx = {k: v - self.n_ready_elements for k, v in self._agent_last_idx.items()}
        self.n_ready_elements = 0

        return ready_elements

    def _finish_path(self) -> None:
        for agent_name, i in self._agent_last_idx.items():
            e = self._all_elements[i]
            e.terminal_dict[agent_name] = self.end_of_episode
            e.next_agent_obs_dict[agent_name] = e.agent_obs_dict[agent_name]

        tick_bound = self.env.tick - self._reward_eval_delay
        i = self.n_ready_elements
        while i < len(self._all_elements):
            if self._discard_tail_elements and self._all_elements[i].tick > tick_bound:
                break

            self._calc_reward_for_element(self._all_elements[i])
            i += 1
        self.n_ready_elements = i
        self._all_elements[self.n_ready_elements :] = []  # Discard tail elements

    def _append_element(self, element: CacheElement) -> None:
        cur_idx = len(self._all_elements)
        self._all_elements.append(element)

        # Update `terminal_dict` & `next_agent_obs_dict`
        for agent_name in element.agent_names:
            self._agent_last_idx[agent_name] = cur_idx
        for agent_name, agent_obs in self.agent_obs_dict.items():
            if agent_name in self._agent_last_idx:
                i = self._agent_last_idx[agent_name]
                e = self._all_elements[i]
                e.terminal_dict[agent_name] = False
                e.next_agent_obs_dict[agent_name] = agent_obs

    def _calc_reward_for_element(self, element: CacheElement) -> None:
        element.reward_dict = self.get_reward(
            event=element.event,
            act_dict=element.action_dict,
            tick=element.tick,
        )
        self.post_step(element)

    def _calc_rewards(self) -> None:
        tick_bound = self.env.tick - self._reward_eval_delay
        i = self.n_ready_elements
        while i < len(self._all_elements) and self._all_elements[i].tick <= tick_bound:
            self._calc_reward_for_element(self._all_elements[i])
            i += 1
        self.n_ready_elements = i

    def _reset(self) -> None:
        self.cur_ep_length = 0
        self.env.reset()
        self._all_elements = []
        self._agent_last_idx = {}
        self.n_ready_elements = 0

        _, self.event, self.end_of_episode = self.env.step(None)
        self.obs, self.agent_obs_dict = self._extract_obs()

    def _step(self, policy_act_dict: Dict[Any, PolicyActionType]) -> None:
        self._total_num_interaction += 1
        self.cur_ep_length += 1

        act_dict = self.policy_act_to_act(self.event, policy_act_dict)
        cache_element = CacheElement(
            event=self.event,
            tick=self.env.tick,
            obs=self.obs,
            agent_obs_dict=self.agent_obs_dict,
            action_dict=act_dict,
            policy_action_dict=policy_act_dict,
            # Following parts will be generated/updated later
            reward_dict={},
            terminal_dict={},
            next_obs=None,
            next_agent_obs_dict={},
            truncated=False,  # TODO: handle this in collector?
        )

        _, self.event, self.end_of_episode = self.env.step(list(act_dict.values()))
        self.obs, self.agent_obs_dict = self._extract_obs()

        cache_element.next_obs = self.obs
        self._append_element(cache_element)

    def _extract_obs(self) -> Tuple[ObservationType, Dict[Any, ObservationType]]:
        if self.end_of_episode:
            return None, {}
        else:
            return self.state_to_obs(self.event, self.env.tick)

    def step(
        self,
        policy_act_dict: Optional[Dict[Any, PolicyActionType]],
    ) -> EnvStepRes:
        if policy_act_dict is not None:
            self._step(policy_act_dict)
        else:
            self._reset()  # Reset

        self._calc_rewards()
        if self.end_of_episode:
            self._finish_path()

        return EnvStepRes(
            tick=self.env.tick,
            event=self.event,
            obs=self.obs,
            agent_obs_dict=self.agent_obs_dict,
            end_of_episode=self.end_of_episode,
        )

    @abstractmethod
    def state_to_obs(self, event: StateType, tick: int = None) -> Tuple[ObservationType, Dict[Any, ObservationType]]:
        raise NotImplementedError

    @abstractmethod
    def policy_act_to_act(
        self,
        event: StateType,
        policy_act_dict: Dict[Any, PolicyActionType],
        tick: int = None,
    ) -> Dict[Any, ActionType]:
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, event: StateType, act_dict: Dict[Any, ActionType], tick: int) -> Dict[Any, float]:
        raise NotImplementedError

    @abstractmethod
    def gather_info(self) -> dict:
        raise NotImplementedError

    def post_step(self, element: CacheElement) -> None:
        pass
