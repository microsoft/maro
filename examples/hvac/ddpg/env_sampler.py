# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import numpy as np

from maro.rl.exploration import MultiLinearExplorationScheduler
from maro.rl.learning import AbsEnvSampler
from maro.rl.policy import DDPG
from maro.simulator import Env
from maro.simulator.scenarios.hvac.common import Action

from .callbacks import baseline
from .config import ac_net_config, ddpg_config, env_config, reward_config, state_config
from .policies import AhuACNet, relative_gaussian_noise

attributes = state_config["attributes"]
agent_name = "ahu01"


class HVACEnvSampler(AbsEnvSampler):
    def __init__(self, get_env, get_policy_func_dict, agent2policy, get_test_env=None, reward_eval_delay=0, parallel_inference=False):
        super().__init__(get_env, get_policy_func_dict, agent2policy, get_test_env=get_test_env, reward_eval_delay=reward_eval_delay, parallel_inference=parallel_inference)
        self._statistics = {
            key: {
                "mean": np.mean(baseline[key]),
                "min": np.min(baseline[key]),
                "max": np.max(baseline[key]),
                "range": np.max(baseline[key]) - np.min(baseline[key]),
            }
            for key in baseline.keys()
        }

    def get_state(self, tick: int = None) -> dict:
        if tick is not None:
            assert tick == self.env.tick
        else:
            tick = self.env.tick

        state = self.env.snapshot_list["ahus"][tick:0:attributes]

        if state_config["normalize"]:
            for i, key in enumerate(attributes):
                state[i] = (state[i] - self._statistics[key]["min"]) / self._statistics[key]["range"]

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

        diff_sps = abs(get_attribute("sps") - get_attribute("sps", t=tick))
        diff_das = abs(get_attribute("das") - get_attribute("das", t=tick))

        efficiency_ratio = abs(get_attribute("kw") / get_attribute("at"))

        reward = -5

        if reward_config["type"] == "Bonsai":
            # Align with Bonsai
            reward = -5
            if diff_sps <= 0.3:
                reward = (
                    math.exp(-efficiency_ratio)
                    - 0.2 * diff_das
                    - 0.05 * (
                        max(0, get_attribute("mat") - 68)   # mat is better to <= 68
                        * max(0, get_attribute("dat") - 57) # dat is better to <= 57
                    )
                )

        elif reward_config["type"] == "V2":
            reward = (
                reward_config["V2_efficiency_factor"] * math.exp(-efficiency_ratio)
                + reward_config["V2_das_diff_factor"] * diff_das
                + reward_config["V2_sps_diff_factor"] * diff_sps
                + reward_config["V2_constraints_factor"] * (
                    max(0, get_attribute("mat") - 68)   # mat is better to <= 68
                    * max(0, get_attribute("dat") - 57) # dat is better to <= 57
                )
            )

            if reward_config["V2_lower_bound"] is not None:
                reward = max(reward, reward_config["V2_lower_bound"])

        elif reward_config["type"] == "V3":
            # The one Lei used
            if diff_sps <= 0.3:
                reward = (
                    2 * (self._statistics["kw"]["max"] - get_attributes("kw")) / self._statistics["kw"]["range"]
                    + (get_attributes("at") - self._statistics["at"]["min"]) / self._statistics["at"]["range"]
                    - 0.05 * diff_das
                    + 10 * (
                        np.max(0.0, 0.05 * (get_attributes("mat") - self._statistics["mat"]["mean"]))
                        * np.min(0.0, self._statistics["mat"]["mean"] - 11 - get_attributes("dat"))
                    )
                )
            else:
                reward = reward_config["V3_threshold"]

        return {agent_name: reward}

    def post_step(self, state, action, env_actions, reward, tick):
        if "reward" not in self.tracker:
            self.tracker["reward"] = []
        self.tracker["reward"].append(reward[agent_name][0] if isinstance(reward[agent_name], np.ndarray) else reward[agent_name])
        if tick == self.env._start_tick + self.env._durations - 1:
            for attribute in ["kw", "at", "dat", "mat"] + ["sps", "das"]:
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
