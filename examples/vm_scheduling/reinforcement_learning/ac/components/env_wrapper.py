# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import scipy.signal
import numpy as np
from collections import defaultdict

from maro.rl import ExperienceSet
from examples.vm_scheduling.reinforcement_learning.common import VMEnvWrapper


class VMEnvWrapperForAC(VMEnvWrapper):
    def get_experiences(self):
        buf = self._replay_buffer

        rewards = np.array(buf["rewards"])
        cumsum_rewards = self.discount_cumsum(rewards, self._gamma)

        exp_set = ExperienceSet(
            buf["states"][:-1],
            buf["actions"][:-1],
            cumsum_rewards[:-1],
            buf["states"][1:],
            buf["info"][1:],
        )
        del buf["states"][:-1]
        del buf["actions"][:-1]
        del buf["rewards"][:-1]
        del buf["info"][:-1]

        return exp_set

    @staticmethod
    def discount_cumsum(x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
            vector x,
            [x0,
            x1,
            x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
