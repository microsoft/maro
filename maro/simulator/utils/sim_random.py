# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from random import Random
from typing import Dict
from collections import OrderedDict


class SimRandom:
    """Simulator random object that will keep a module level random.Random object to keep its internal random sequence, to make sure that,
    it will not be affected by outside, and outside can set seed with seed function
    """
    def __init__(self):
        # random object instances
        self._rand_instances: Dict[str, Random] = OrderedDict()
        self._seed_dict: Dict[str, int] = {}
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
        for key, rand in self._rand_instances.items():
            # we set seed for each random instance with 1 offset
            seed = seed_num + self._index

            rand.seed(seed)
            
            self._seed_dict[key] = seed

            self._index += 1

    def __getitem__(self, key):
        assert type(key) is str

        if key not in self._rand_instances:
            r = Random()
            r.seed(self._seed + self._index)

            self._index += 1

            self._rand_instances[key] = r

        return self._rand_instances[key]

    def get_seed(self, key: str = None) -> int:
        """Get seed of current random generator
        
        Args:
            key(str): key of item to get

        Returns:
            int: if key is None return seed for 1st instance (same as what passed to seed function), else return seed for specified generator
        """
        if key is not None:
            return self._seed_dict.get(key, None)

        return self._seed


random = SimRandom()
"""random utility for simulator, same with original random module """

seed = random.seed
"""set seed for simulator"""

__all__ = ['seed', 'random', 'SimRandom']
