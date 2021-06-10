# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pdb
import numpy as np
from collections import defaultdict

from maro.rl import ExperienceSet
from maro.simulator import Env

from examples.vm_scheduling.reinforcement_learning.common import VMEnvWrapper


class VMEnvWrapperForDQN(VMEnvWrapper):
    def __init__(
        self,
        env,
        training: bool,
        alpha: float,
        beta: float,
        pm_num: int,
        durations: int,
        vm_window_size: int,
        pm_window_size: int,
        gamma: float,
        window_type: str,
        window_size: int,
    ):
        super().__init__(
            env, training, alpha, beta, pm_num, durations,
            vm_window_size, pm_window_size, gamma
        )
        self._window_type = window_type # determine the type of the used window (fix or flexible)
        self._window_size = window_size # the size of the used window

    def get_experiences(self):
        buf = self._replay_buffer
        exp = defaultdict(list)

        if self._window_type == "fix":
            for i in range(len(buf["states"]) - self._window_size):
                st, en = i, i + self._window_size
                exp["states"].append(buf["states"][st])
                exp["actions"].append(buf["actions"][st])

                reward, gamma = 0.0, 1.0
                for j in range(en - 1, st - 1, -1):
                    reward = (
                        reward * gamma
                        + buf["rewards"][j]
                    )
                    gamma *= self._gamma

                if self._gamma != 1.0:
                    store_gamma = gamma
                else:
                    store_gamma = 0.0
                exp["rewards"].append(reward)

                exp["next_states"].append(buf["states"][en])
                info = np.zeros(buf["info"][en].shape[0] + 1)
                info[:-1] = buf["info"][en]
                info[-1] = store_gamma
                exp["info"].append(info)
        else:
            st = 0
            while st < len(buf["states"]) :
                if st + self._window_size >= len(buf["states"]):
                    en = len(buf["states"]) - 1
                else:
                    en = st + self._window_size

                reward, gamma = 0.0, 1.0

                for i in range(en-1, st-1, -1):
                    exp["states"].append(buf["states"][i])
                    exp["actions"].append(buf["actions"][i])

                    reward = (
                        reward * gamma
                        + buf["rewards"][i]
                    )
                    gamma *= self._gamma

                    if self._gamma != 1.0:
                        store_gamma = gamma
                    else:
                        store_gamma = 0.0
                    exp["rewards"].append(reward)

                    exp["next_states"].append(buf["states"][en])
                    info = np.zeros(buf["info"][en].shape[0] + 1)
                    info[:-1] = buf["info"][en]
                    info[-1] = store_gamma
                    exp["info"].append(info)

                st += self._window_size

        exp_set = ExperienceSet(
            exp["states"],
            exp["actions"],
            exp["rewards"],
            exp["next_states"],
            exp["info"]
        )
        del buf["states"][:-1]
        del buf["actions"][:-1]
        del buf["rewards"][:-1]
        del buf["info"][:-1]

        return exp_set
