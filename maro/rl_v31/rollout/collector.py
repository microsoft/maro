# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from tianshou.data import Batch

from maro.rl_v31.objects import ExpElement
from maro.rl_v31.rollout.typing import EnvStepRes
from maro.rl_v31.rollout.venv import BaseVectorEnv
from maro.rl_v31.rollout.wrapper import AgentWrapper

# TODO: handle truncated


class Collector(object):
    def __init__(
        self,
        venv: BaseVectorEnv,
        agent_wrapper: AgentWrapper,
    ) -> None:
        self._venv = venv
        self._data: List[EnvStepRes] = [EnvStepRes.dummy() for _ in range(self.env_num)]
        self._agent_wrapper = agent_wrapper

    @property
    def env_num(self) -> int:
        return self._venv.env_num

    def _update_data(self, res_dict: Dict[int, EnvStepRes]) -> None:
        for i, step_res in res_dict.items():
            self._data[i] = step_res

    def switch_explore(self, explore: bool) -> None:
        self._agent_wrapper.switch_explore(explore)

    def reset(self, **kwargs: Any) -> None:
        self._data: List[EnvStepRes] = [EnvStepRes.dummy() for _ in range(self.env_num)]

    def reset_envs(self, ids: Optional[List[int]] = None, **kwargs: Any) -> None:
        ids = list(range(self.env_num)) if ids is None else ids
        self._update_data(self._venv.reset(ids, **kwargs))

    def collect(
        self,
        n_steps: Optional[int] = None,
        n_episodes: Optional[int] = None,
        policy_state: Optional[dict] = None,  # TODO: check if this is needed
    ) -> Tuple[List[dict], Dict[int, List[ExpElement]]]:
        assert any(
            [
                n_steps is None and n_episodes is not None,
                n_steps is not None and n_episodes is None,
            ],
        ), "Please provide exactly one of n_steps and n_episodes"

        if policy_state is not None:
            self._agent_wrapper.set_policy_state(policy_state)

        total_infos, env_exps = (
            self._collect_n_steps(n_steps) if n_steps is not None else self._collect_n_episodes(n_episodes)
        )
        return total_infos, env_exps

    def _collect_n_steps(self, n_steps: int) -> Tuple[List[dict], Dict[int, List[ExpElement]]]:
        assert n_steps > 0

        env_ids = list(range(self.env_num))

        total_exps: Dict[int, List[ExpElement]] = {i: [] for i in env_ids}
        while n_steps > 0:
            reset_ids = [i for i in env_ids if self._data[i].end_of_episode]
            self.reset_envs(reset_ids)

            # Compose
            agent_state_agg = defaultdict(list)
            for i in env_ids:
                for agent_name, agent_obs in self._data[i].agent_obs_dict.items():
                    agent_state_agg[agent_name].append(Batch(obs=agent_obs))

            # Get actions
            batch_by_agent = {agent_name: Batch.stack(obs) for agent_name, obs in agent_state_agg.items()}
            policy_act_dict = self._agent_wrapper.choose_actions(batch_by_agent)

            # Decompose
            env_policy_acts = {}
            for i in env_ids:
                cur = {agent_name: act[i] for agent_name, act in policy_act_dict.items()}
                env_policy_acts[i] = cur

            self._update_data(self._venv.step(env_policy_acts))
            env_ready_elements = self._venv.collect_ready_exps(env_ids)

            for i, ready_elements in env_ready_elements.items():
                total_exps[i] += ready_elements
                n_steps -= len(ready_elements)

        total_infos: List[dict] = list(
            self._venv.gather_info(env_ids).values(),
        )  # TODO: shall we gather info everytime a episode finishes?
        return total_infos, total_exps

    def _collect_n_episodes(self, n_episodes: int) -> Tuple[List[dict], Dict[int, List[ExpElement]]]:
        assert n_episodes > 0

        waiting_env_ids = deque([i for i in range(self.env_num) if self._data[i].end_of_episode])
        assert len(waiting_env_ids) > 0, "No usable envs. Stop data collections."

        env_ids = []
        total_exps: Dict[int, List[ExpElement]] = defaultdict(list)
        total_infos: List[dict] = []
        while n_episodes > 0 or len(env_ids) > 0:
            # Recycle finished envs
            recycle_env_ids = [i for i in env_ids if self._data[i].end_of_episode]
            if len(recycle_env_ids) > 0:
                infos = self._venv.gather_info(recycle_env_ids)
                total_infos += list(infos.values())

                waiting_env_ids.extend(recycle_env_ids)

                recycle_env_ids = set(recycle_env_ids)
                env_ids = [e for e in env_ids if e not in recycle_env_ids]

            # Allocate new envs if needed
            m = min(n_episodes, len(waiting_env_ids))
            if m > 0:
                allocate_env_ids = [waiting_env_ids.popleft() for _ in range(m)]
                env_ids += allocate_env_ids
                n_episodes -= m
                self.reset_envs(allocate_env_ids)

            # Compose
            agent_state_agg = defaultdict(list)
            for i in env_ids:
                for agent_name, agent_obs in self._data[i].agent_obs_dict.items():
                    agent_state_agg[agent_name].append(Batch(obs=agent_obs))

            # Get actions
            batch_by_agent = {agent_name: Batch.stack(obs) for agent_name, obs in agent_state_agg.items()}
            policy_act_dict = self._agent_wrapper.choose_actions(batch_by_agent)

            # Decompose
            env_policy_acts = {}
            for i, e in enumerate(env_ids):
                cur = {agent_name: act[i] for agent_name, act in policy_act_dict.items()}
                env_policy_acts[e] = cur

            self._update_data(self._venv.step(env_policy_acts))
            env_ready_elements = self._venv.collect_ready_exps(list(env_ids))

            for i, ready_elements in env_ready_elements.items():
                total_exps[i] += ready_elements

        return total_infos, total_exps
