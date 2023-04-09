from typing import Any, Dict, Optional, Tuple, cast

import numpy as np

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

        self._rewards = []

    def state_to_obs(self, event: DecisionEvent, tick: int = None) -> Tuple[None, Dict[int, np.ndarray]]:
        return None, {0: event.state}

    def policy_act_to_act(
        self,
        event: DecisionEvent,
        policy_act_dict: Dict[int, np.ndarray],
        tick: int = None,
    ) -> Dict[int, Action]:
        return {k: Action(v) for k, v in policy_act_dict.items()}

    def get_reward(self, event: DecisionEvent, act_dict: Dict[int, np.ndarray], tick: int) -> Dict[int, float]:
        be = cast(GymBusinessEngine, self.env.business_engine)
        return {0: be.get_reward_at_tick(tick)}

    def gather_info(self) -> dict:
        pass
