# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import numpy as np

from maro.rl.exploration import MultiLinearExplorationScheduler
from maro.rl.learning import AbsEnvSampler
from maro.rl.policy import DDPG
from maro.simulator import Env
from maro.simulator.scenarios.hvac.common import Action

from .config import ac_net_config, ddpg_config, env_config, state_config
from .policies import AhuACNet, relative_gaussian_noise

attributes = state_config["attributes"]
agent_name = "ahu01"


class HVACEnvSampler(AbsEnvSampler):
    def __init__(self, get_env, get_policy_func_dict, agent2policy, get_test_env=None, reward_eval_delay=0, parallel_inference=False):
        super().__init__(get_env, get_policy_func_dict, agent2policy, get_test_env=get_test_env, reward_eval_delay=reward_eval_delay, parallel_inference=parallel_inference)
        # from .callbacks import baseline
        # self._sps = baseline["sps"]
        # self._das = baseline["das"]

    def get_state(self, tick: int = None) -> dict:
        if tick is not None:
            assert tick == self.env.tick
        else:
            tick = self.env.tick
        state = self.env.snapshot_list["ahus"][tick:0:attributes]
        return {agent_name: state}

    def get_env_actions(self, action: dict) -> list:
        return [
            Action(ahu_idx=0, sps=act[0], das=act[1])
            for act in action.values()
        ]

        # return [Action(ahu_idx=0, sps=self._sps[self.env.tick], das=self._das[self.env.tick])]

    def get_reward(self, actions: list, tick: int):
        def get_attribute(name: str, idx: int=0, t: int=tick+1):
            return self.env.snapshot_list["ahus"][t:idx:name]

        """
        # Align with Bonsai
        reward = -5
        diff_sps = abs(get_attribute("sps") - get_attribute("sps", t=tick))

        if diff_sps <= 0.3:
            efficiency_ratio = abs(get_attribute("kw") / get_attribute("at"))
            diff_das = abs(get_attribute("das") - get_attribute("das", t=tick))

            reward = (
                math.exp(-efficiency_ratio)
                - 0.2 * diff_das
                - 0.05 * (
                    max(0, get_attribute("mat") - 68)   # mat is better to <= 68
                    * max(0, get_attribute("dat") - 57) # dat is better to <= 57
                )
            )
        """

        diff_sps = abs(get_attribute("sps") - get_attribute("sps", t=tick))
        diff_das = abs(get_attribute("das") - get_attribute("das", t=tick))

        efficiency_ratio = abs(get_attribute("kw") / get_attribute("at"))

        reward = (
            10 * math.exp(-efficiency_ratio)
            - 2 * diff_das
            - 5 * diff_sps
            - 0.5 * (
                max(0, get_attribute("mat") - 68)   # mat is better to <= 68
                * max(0, get_attribute("dat") - 57) # dat is better to <= 57
            )
        )

        reward = max(reward, -2.5)

        return {agent_name: reward}

    def post_step(self, state, action, env_actions, reward, tick):
        if "reward" not in self.tracker:
            self.tracker["reward"] = []
        self.tracker["reward"].append(reward[agent_name][0] if isinstance(reward[agent_name], np.ndarray) else reward[agent_name])
        if tick == self.env._start_tick + self.env._durations - 1:
            for attribute in attributes + ["sps", "das"]:
                self.tracker[attribute] = self.env.snapshot_list["ahus"][::attribute][1:]
            self.tracker["total_kw"] = np.cumsum(self.tracker["kw"])
            self.tracker["reward"] = self.tracker["reward"][1:]
            self.tracker["total_reward"] = np.cumsum(self.tracker["reward"])


policy_func_dict = {
    "ddpg": lambda name: DDPG(
        name=name,
        ac_net=AhuACNet(**ac_net_config),
        reward_discount=0,
        warmup=0,            # ?: Is 5000 reasonable?
        exploration_strategy=(relative_gaussian_noise, ddpg_config["exploration_strategy"]),
        exploration_scheduling_options=[
            ("mean", MultiLinearExplorationScheduler, ddpg_config["exploration_mean_scheduler_options"]),
        ],
    )
}

def get_env_sampler():
    return HVACEnvSampler(
        get_env=lambda: Env(scenario="hvac", **env_config),
        get_policy_func_dict=policy_func_dict,
        agent2policy={agent_name: "ddpg"},
        get_test_env=lambda: Env(scenario="hvac", **env_config)
    )
