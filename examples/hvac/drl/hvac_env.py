# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import gym
import numpy as np
from gym import spaces

from maro.simulator import Env
from maro.simulator.scenarios.hvac.common import Action
from examples.hvac.rl.config import env_config
from examples.hvac.rl.callbacks import baseline


class MAROHAVEnv(gym.Env):
    id = "HVA Controller"

    def __init__(self):
        self.env = Env(scenario="hvac", **env_config)

        self._statistics = {
            key: {
                "mean": np.mean(baseline[key]),
                "min": np.min(baseline[key]),
                "max": np.max(baseline[key]),
                "range": np.max(baseline[key]) - np.min(baseline[key]),
            }
            for key in baseline.keys()
        }

        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

        self._max_episode_steps = 500

    def step(self, action):
        act = Action(
            ahu_idx=0,
            sps=0.6 + (action[0] + 1) * (1.1 - 0.6) / 2,
            das=45 + (action[1] + 1) * (60 - 45) / 2
        )

        _, _, is_done = self.env.step(act)

        return self.get_state(), self.get_reward(), is_done, {}

    def get_state(self):
        attributes = ["kw", "at", "dat", "mat"]
        state = self.env.snapshot_list["ahus"][self.env.tick:0:attributes]
        for idx, att in enumerate(attributes):
            state[idx] = (state[idx] - self._statistics[att]["min"]) / self._statistics[att]["range"]
        return state

    def get_reward(self):
        def attribute(name: str, t: int):
            return self.env.snapshot_list["ahus"][t:0:name]

        tick = self.env.tick
        diff_sps = abs(attribute("sps", tick) - attribute("sps", tick - 1))
        diff_das = abs(attribute("das", tick) - attribute("das", tick - 1))

        if diff_sps <= 0.3:
            reward = (
                2 * (self._statistics["kw"]["max"] - attribute("kw", tick)) / self._statistics["kw"]["range"]
                + (attribute("at", tick) - self._statistics["at"]["min"]) / self._statistics["at"]["range"]
                - 0.05 * diff_das
                + (
                    10
                    * max(0, 0.05 * (attribute("mat", tick) - self._statistics["mat"]["mean"]))
                    * min(0, self._statistics["dat"]["mean"] - attribute("dat", tick))
                )
            )
        else:
            reward = -5

        return float(reward)

    def reset(self):
        self.env.reset()
        self.env.step(None)
        return self.get_state()
