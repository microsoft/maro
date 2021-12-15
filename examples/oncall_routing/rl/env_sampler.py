# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Dict, List

import numpy as np

from maro.rl.learning import AbsEnvSampler
from maro.simulator import Env
from maro.simulator.scenarios.oncall_routing import Action, OncallRoutingPayload

from examples.oncall_routing.rl.state_shaping_utils import get_state_info_dict

agent_id = "high-level"


class OncallEnvSampler(AbsEnvSampler):
    # TODO: (by Jinyu) update the interface after the workflow refinement done.
    def __init__(
        self,
        get_env: Callable[[], Env],
        get_policy_func_dict: Dict[str, Callable],
        agent2policy: Dict[str, str],
        get_test_env: Callable[[], Env] = None,
        reward_eval_delay: int = 0,
        parallel_inference: bool = False
    ):
        super().__init__(
            get_env,
            get_policy_func_dict,
            agent2policy,
            get_test_env=get_test_env,
            reward_eval_delay=reward_eval_delay,
            parallel_inference=parallel_inference
        )

    def get_state(self, tick: int = None) -> Dict[str, np.ndarray]:
        """Compute the state for a given tick.

        Args:
            tick (int): The tick for which to compute the environmental state. If computing the current state,
                use tick=self.env.tick.
        Returns:
            A dictionary with (agent ID, state) as key-value pairs.
        """
        assert isinstance(self.event, OncallRoutingPayload)

        if tick == None:
            tick = self.env.tick

        state_info_dict = get_state_info_dict(tick, self.event, self.env.snapshot_list)
        # TODO: (according to the policy) generate the state for policy with the state_info_dict
        # state = ...

        return {
            agent_id: [] # TODO: replate with the state generated above
        }

    def get_env_actions(self, action: List[np.ndarray]) -> List[Action]:
        """Convert policy outputs to an action that can be executed by ``self.env.step()``."""
        # TODO: add the convert logic.
        return []

    def get_reward(self, actions: List[Action], tick: int):
        """Evaluate the reward for an action.
        Args:
            tick (int): Evaluate the reward for the actions that occured at the given tick. Each action in
                ``actions`` must be an Action object defined for the environment in question.

        Returns:
            A dictionary with (agent ID, reward) as key-value pairs.
        """
        return {
            agent_id: 0  # TODO: replace the value with a meaning reward.
        }
