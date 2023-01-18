# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any, Dict, Tuple, Union

import numpy as np

from maro.rl.rollout import AbsEnvSampler, CacheElement
from maro.simulator.scenarios.gym.business_engine import GymBusinessEngine
from maro.simulator.scenarios.gym.common import Action


def _show_info(rewards: list, tag: str) -> None:
    print(
        f"[{tag}] Total N-steps = {sum([len(e) for e in rewards])}, "
        f"N segments = {len(rewards)}, "
        f"Average reward = {np.mean([sum(e) for e in rewards]):.4f}, "
        f"Max reward = {np.max([sum(e) for e in rewards]):.4f}, "
        f"Min reward = {np.min([sum(e) for e in rewards]):.4f}, "
        f"Average N-steps = {np.mean([len(e) for e in rewards]):.1f}\n",
    )


class GymEnvSampler(AbsEnvSampler):
    def _get_global_and_agent_state_impl(
        self,
        event: np.ndarray,
        tick: int = None,
    ) -> Tuple[Union[None, np.ndarray, list], Dict[Any, Union[np.ndarray, list]]]:
        return None, {0: event.state}

    def _translate_to_env_action(self, action_dict: dict, event: Any) -> dict:
        return {k: Action(v) for k, v in action_dict.items()}

    def _get_reward(self, env_action_dict: dict, event: Any, tick: int) -> Dict[Any, float]:
        be = self._env.business_engine
        assert isinstance(be, GymBusinessEngine)
        return {0: be.get_reward_at_tick(tick)}

    def _post_step(self, cache_element: CacheElement) -> None:
        self._info["env_metric"] = self._env.metrics

    def _post_eval_step(self, cache_element: CacheElement) -> None:
        self._post_step(cache_element)

    def post_collect(self, info_list: list, ep: int) -> None:
        rewards = [list(e["env_metric"]["reward_record"].values()) for e in info_list]
        _show_info(rewards, "Collect")

    def post_evaluate(self, info_list: list, ep: int) -> None:
        rewards = [list(e["env_metric"]["reward_record"].values()) for e in info_list]
        _show_info(rewards, "Evaluate")
