from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from maro.rl.policy import AbsPolicy
from maro.rl.rollout import AbsEnvSampler, CacheElement
from maro.simulator import Env
from maro.simulator.scenarios.gym.business_engine import GymBusinessEngine

from .config import algorithm
from .env_helper import env_conf, helper_env


class GymEnvSampler(AbsEnvSampler):
    def _get_global_and_agent_state_impl(
        self,
        event: np.ndarray,
        tick: int = None,
    ) -> Tuple[Optional[np.ndarray], Dict[Any, np.ndarray]]:
        return None, {0: event}

    def _translate_to_env_action(self, action_dict: Dict[Any, np.ndarray], event: object) -> Dict[Any, object]:
        return action_dict  # TODO

    def _get_reward(self, env_action_dict: Dict[Any, object], event: object, tick: int) -> Dict[Any, float]:
        be = self._env.business_engine
        assert isinstance(be, GymBusinessEngine)
        return {0: be.get_reward_at_tick(tick)}

    def _post_step(self, cache_element: CacheElement, reward: Dict[Any, float]) -> None:
        self._info["env_metric"] = self._env.metrics

    def _post_eval_step(self, cache_element: CacheElement, reward: Dict[Any, float]) -> None:
        self._post_step(cache_element, reward)


agent2policy = {agent: f"{algorithm}_{agent}.policy" for agent in helper_env.agent_idx_list}


def env_sampler_creator(policy_creator: Dict[str, Callable[[str], AbsPolicy]]) -> GymEnvSampler:
    return GymEnvSampler(
        get_env=lambda: Env(**env_conf),
        policy_creator=policy_creator,
        agent2policy=agent2policy,
    )
