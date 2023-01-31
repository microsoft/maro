# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, Tuple, Union

import numpy as np

from maro.rl.rollout import AbsEnvSampler, CacheElement

from tests.rl.gym_wrapper.simulator.business_engine import GymBusinessEngine
from tests.rl.gym_wrapper.simulator.common import Action, DecisionEvent


def _calc_metrics(rewards: list) -> dict:
    return {
        "n_step": sum([len(e) for e in rewards]),
        "n_segment": len(rewards),
        "avg_reward": np.mean([sum(e) for e in rewards]),
        "max_reward": np.max([sum(e) for e in rewards]),
        "min_reward": np.min([sum(e) for e in rewards]),
        "avg_n_step": np.mean([len(e) for e in rewards]),
    }


class GymEnvSampler(AbsEnvSampler):
    def _get_global_and_agent_state_impl(
        self,
        event: DecisionEvent,
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
        metrics = _calc_metrics(rewards)
        self.metrics.update(metrics)

    def post_evaluate(self, info_list: list, ep: int) -> None:
        rewards = [list(e["env_metric"]["reward_record"].values()) for e in info_list]
        metrics = _calc_metrics(rewards)
        self.metrics.update({"val/" + k: v for k, v in metrics.items()})
