from typing import Dict, Optional, Tuple, cast

import numpy as np
from maro.rl_v31.objects import CacheElement

from maro.rl_v31.rollout.wrapper import EnvWrapper
from maro.simulator import Env
from .simulator.business_engine import GymBusinessEngine
from .simulator.common import Action, DecisionEvent


class GymEnvWrapper(EnvWrapper):
    def __init__(
        self,
        env: Env,
        reward_eval_delay: Optional[int] = None,
        max_episode_length: Optional[int] = None,
        discard_tail_elements: bool = False,
    ) -> None:
        super().__init__(
            env=env,
            reward_eval_delay=reward_eval_delay,
            max_episode_length=max_episode_length,
            discard_tail_elements=discard_tail_elements,
        )

        self._reward_history = []
        self._metrics = {}

    def state_to_obs(self, event: DecisionEvent, tick: int = None) -> Tuple[None, Dict[int, np.ndarray]]:
        return None, {0: event.state}

    def policy_act_to_act(
        self,
        event: DecisionEvent,
        policy_act_dict: Dict[int, np.ndarray],
        tick: int = None,
    ) -> Dict[int, Action]:
        return {k: Action(v.numpy()) for k, v in policy_act_dict.items()}

    def get_reward(self, event: DecisionEvent, act_dict: Dict[int, np.ndarray], tick: int) -> Dict[int, float]:
        be = cast(GymBusinessEngine, self.env.business_engine)
        return {0: be.get_reward_at_tick(tick)}

    def gather_info(self) -> dict:
        if len(self._reward_history) > 0:
            cur = {
                "n_steps": sum([n for n, _ in self._reward_history]),
                "n_segment": len(self._reward_history),
                "avg_reward": np.mean([r for _, r in self._reward_history]),
                "avg_n_steps": np.mean([n for n, _ in self._reward_history]),
                "max_n_steps": np.max([n for n, _ in self._reward_history]),
                "n_interactions": self._total_num_interaction,
            }
            self._reward_history.clear()
            return cur
        else:
            return {"n_interactions": self._total_num_interaction}

    def post_step(self, element: CacheElement) -> None:
        if not (self.end_of_episode or element.truncated):
            return

        cur_metrics = list(self.env.metrics["reward_record"].values())
        self._reward_history.append((len(cur_metrics), np.sum(cur_metrics)))
