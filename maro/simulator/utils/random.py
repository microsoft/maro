# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


"""
This module will keep a module level random.Random object to keep its internal random sequence, to make sure that,
it will not be affected by outside, and outside can set seed with seed function
"""

import time
from random import Random
from typing import Dict
from collections import OrderedDict


class SimRandom:
    def __init__(self):
        # random object instances
        self._rand_instances: Dict[str, Random] = OrderedDict()
        self._seed = int(time.time())
        self._index = 0

    def seed(self, seed_num: int):
        """Set seed for simulator

        Args:
            seed_num (int): seed to set, must be an integer
        """
        assert type(seed_num) is int

        self._seed = seed_num

        self._index = 0
        for _, rand in self._rand_instances.items():
            # we set seed for each random instance with 1 offset
            rand.seed(seed_num + self._index)

            self._index += 1

    def __getitem__(self, key):
        assert type(key) is str

        if key not in self._rand_instances:
            r = Random()
            r.seed(self._seed + self._index)

            self._index += 1

            self._rand_instances[key] = r

        return self._rand_instances[key]


random = SimRandom()

seed = random.seed

__all__ = ['seed', 'random']
