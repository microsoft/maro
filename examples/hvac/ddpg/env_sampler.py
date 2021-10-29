# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import numpy as np

from maro.rl.learning import AbsEnvSampler
from maro.rl.policy import DDPG
from maro.simulator import Env
from maro.simulator.scenarios.hvac.common import Action

from .config import ac_net_config, env_config, state_config
from .policies import AhuACNet

attributes = state_config["attributes"]
agent_name = "ahu01"


class HVACEnvSampler(AbsEnvSampler):
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

    def get_reward(self, actions: list, tick: int):
        def get_attribute(name: str, idx: int=0, t: int=tick+1):
            return self.env.snapshot_list["ahus"][t:idx:name]

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

        return {agent_name: reward}

    def post_step(self, state, action, env_actions, reward, tick):
        if self.env.tick == self.env._start_tick + self.env._durations - 1:
            for attribute in attributes + ["sps", "das"]:
                self.tracker[attribute] = self.env.snapshot_list["ahus"][::attribute][1:]
            self.tracker["total_kw"] = np.cumsum(self.tracker["kw"])


policy_func_dict = {
    "ddpg": lambda name: DDPG(
        name=name,
        ac_net=AhuACNet(**ac_net_config),
        reward_discount=0,
        warmup=0            # ?: Is 5000 reasonable?
    )
}

def get_env_sampler():
    return HVACEnvSampler(
        get_env=lambda: Env(scenario="hvac", **env_config),
        get_policy_func_dict=policy_func_dict,
        agent2policy={agent_name: "ddpg"},
        get_test_env=lambda: Env(scenario="hvac", **env_config)
    )
