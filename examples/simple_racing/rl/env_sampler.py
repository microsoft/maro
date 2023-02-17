# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np

from maro.rl.policy.abs_policy import AbsPolicy
from maro.rl.rollout import AbsEnvSampler, CacheElement
from maro.rl.rollout.env_sampler import AbsAgentWrapper, SimpleAgentWrapper
from maro.simulator.core import Env

from maro.simulator.scenarios.simple_racing.business_engine import SimpleRacingBusinessEngine
from maro.simulator.scenarios.simple_racing.common import Action, DecisionEvent


class SimpleRacingEnvSampler(AbsEnvSampler):
    def __init__(
        self,
        learn_env: Env,
        test_env: Env,
        policies: List[AbsPolicy],
        agent2policy: Dict[Any, str],
        trainable_policies: List[str] = None,
        agent_wrapper_cls: Type[AbsAgentWrapper] = SimpleAgentWrapper,
        reward_eval_delay: int = None,
        max_episode_length: int = None,
    ) -> None:
        super(SimpleRacingEnvSampler, self).__init__(
            learn_env=learn_env,
            test_env=test_env,
            policies=policies,
            agent2policy=agent2policy,
            trainable_policies=trainable_policies,
            agent_wrapper_cls=agent_wrapper_cls,
            reward_eval_delay=reward_eval_delay,
            max_episode_length=max_episode_length,
        )

        self._sample_rewards = []
        self._eval_rewards = []

    def _get_global_and_agent_state_impl(
        self,
        event: DecisionEvent,
        tick: int = None,
    ) -> Tuple[Union[None, np.ndarray, list], Dict[Any, Union[np.ndarray, list]]]:
        # TODO: Add your state shaping logic here if you define a new DecisionEvent to maintain richer information.
        # Internal variables defined in the SimpleRacingBusinessEngine can be access from: self._env.business_engine.
        return None, {0: event.state}

    def _translate_to_env_action(self, action_dict: dict, event: Any) -> dict:
        # TODO: Add your action shaping logic here if you define a new Action class to maintain richer information.
        # Internal variables defined in the SimpleRacingBusinessEngine can be access from: self._env.business_engine.
        return {k: Action(v) for k, v in action_dict.items()}

    def _get_reward(self, env_action_dict: dict, event: Any, tick: int) -> Dict[Any, float]:
        # TODO: Add your reward shaping logic here if you want to improve your reward definition.
        # Internal variables defined in the SimpleRacingBusinessEngine can be access from: self._env.business_engine.
        be = self._env.business_engine
        assert isinstance(be, SimpleRacingBusinessEngine)
        return {0: be.get_reward_at_tick(tick)}

    def _post_step(self, cache_element: CacheElement) -> None:
        # TODO: Add here if you have any other metrics after each env step to track in MARO logging.
        if not (self._end_of_episode or self.truncated):
            return
        rewards = list(self._env.metrics["reward_record"].values())
        self._sample_rewards.append((len(rewards), np.sum(rewards)))

    def _post_eval_step(self, cache_element: CacheElement) -> None:
        # TODO: Add here if you have any other metrics after each env step when evaluation to track in MARO logging.
        if not (self._end_of_episode or self.truncated):
            return
        rewards = list(self._env.metrics["reward_record"].values())
        self._eval_rewards.append((len(rewards), np.sum(rewards)))

    def post_collect(self, info_list: list, ep: int) -> None:
        # TODO: Add here if you have any other metrics after each training episode to track in MARO logging.
        cur = {
            "n_steps": sum([n for n, _ in self._sample_rewards]),
            "n_segment": len(self._sample_rewards),
            "avg_reward": np.mean([r for _, r in self._sample_rewards]),
            "avg_n_steps": np.mean([n for n, _ in self._sample_rewards]),
            "max_n_steps": np.max([n for n, _ in self._sample_rewards]),
            "n_interactions": self._total_number_interactions,
        }
        self.metrics.update(cur)
        # clear validation metrics
        self.metrics = {k: v for k, v in self.metrics.items() if not k.startswith("val/")}
        self._sample_rewards.clear()

    def post_evaluate(self, info_list: list, ep: int) -> None:
        # TODO: Add here if you have any other metrics after each evaluation episode to track in MARO logging.
        cur = {
            "val/n_steps": sum([n for n, _ in self._eval_rewards]),
            "val/n_segment": len(self._eval_rewards),
            "val/avg_reward": np.mean([r for _, r in self._eval_rewards]),
            "val/avg_n_steps": np.mean([n for n, _ in self._eval_rewards]),
            "val/max_n_steps": np.max([n for n, _ in self._eval_rewards]),
        }
        self.metrics.update(cur)
        self._eval_rewards.clear()
