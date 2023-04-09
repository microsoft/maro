# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from maro.rl_v31.rollout.typing import ActionType, ObservationType, PolicyActionType


@dataclass
class ExpElement:
    tick: int
    obs: ObservationType
    agent_obs_dict: Dict[Any, ObservationType]
    policy_action_dict: Dict[Any, PolicyActionType]
    reward_dict: Dict[Any, float]
    terminal_dict: Dict[Any, bool]
    next_obs: Optional[ObservationType]
    next_agent_obs_dict: Optional[Dict[Any, ObservationType]]
    truncated: bool

    @property
    def agent_names(self) -> list:
        return sorted(self.agent_obs_dict.keys())

    @property
    def num_agents(self) -> int:
        return len(self.agent_obs_dict)

    def split_contents_by_trainer(self, agent2trainer: Dict[Any, str]) -> Dict[str, ExpElement]:
        ret: Dict[str, ExpElement] = defaultdict(
            lambda: ExpElement(
                tick=self.tick,
                obs=self.obs,
                agent_obs_dict={},  # To be filled
                policy_action_dict={},  # To be filled
                reward_dict={},  # To be filled
                terminal_dict={},  # To be filled
                next_obs=self.next_obs,
                next_agent_obs_dict=None if self.next_agent_obs_dict is None else {},
                truncated=self.truncated,
            )
        )

        for agent_name in self.agent_obs_dict:
            trainer_name = agent2trainer[agent_name]
            ret[trainer_name].agent_obs_dict[agent_name] = self.agent_obs_dict[agent_name]
            ret[trainer_name].policy_action_dict[agent_name] = self.policy_action_dict[agent_name]
            ret[trainer_name].reward_dict[agent_name] = self.reward_dict[agent_name]
            ret[trainer_name].terminal_dict[agent_name] = self.terminal_dict[agent_name]
            if self.next_agent_obs_dict is not None:
                ret[trainer_name].next_agent_obs_dict[agent_name] = self.next_agent_obs_dict[agent_name]

        return ret

    def split_contents_by_agent(self) -> Dict[Any, ExpElement]:
        ret = {}
        for agent_name in self.agent_obs_dict.keys():
            ret[agent_name] = ExpElement(
                tick=self.tick,
                obs=self.obs,
                agent_obs_dict={agent_name: self.agent_obs_dict[agent_name]},
                policy_action_dict={agent_name: self.policy_action_dict[agent_name]},
                reward_dict={agent_name: self.reward_dict[agent_name]},
                terminal_dict={agent_name: self.terminal_dict[agent_name]},
                next_obs=self.next_obs,
                next_agent_obs_dict={
                    agent_name: self.next_agent_obs_dict[agent_name],
                }
                if self.next_agent_obs_dict is not None and agent_name in self.next_agent_obs_dict
                else None,
                truncated=self.truncated,
            )
        return ret


@dataclass
class CacheElement(ExpElement):
    event: Any
    action_dict: Dict[Any, ActionType]

    def to_exp_element(self) -> ExpElement:
        return ExpElement(
            tick=self.tick,
            obs=self.obs,
            agent_obs_dict=self.agent_obs_dict,
            policy_action_dict=self.policy_action_dict,
            reward_dict=self.reward_dict,
            terminal_dict=self.terminal_dict,
            next_obs=self.next_obs,
            next_agent_obs_dict=self.next_agent_obs_dict,
            truncated=self.truncated,
        )
