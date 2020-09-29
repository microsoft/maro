# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from collections import OrderedDict
from random import Random
from typing import Dict


class SimRandom:
    """Simulator random object that will keep a module level random.Random object to keep its internal random sequence,
    it will not be affected by outside, and outside can set seed with seed function as normal.

    Use it as a dict to get another random object with a name, all the random objects from this way will be
    affect the seed method.

    .. code-block:: python

        from maro.simulator.utils import random, seed

        # This will create 2 random object, each has different sequence.
        r1 = random["r1"]
        r2 = random["r2"]

        # Seed will reset above random sequence.
        seed(1)
    """

    def __init__(self):
        # random object instances
        self._rand_instances: Dict[str, Random] = OrderedDict()
        self._seed_dict: Dict[str, int] = {}
        self._seed = int(time.time())
        self._index = 0

    def seed(self, seed_num: int):
        """Set seed for simulator random objects.

        NOTE:
            This method will affect all the random object that get from this class.

        Args:
            seed_num (int): Seed to set, must be an integer.
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
        """Get seed of current random generator.

        NOTE:
            This will only return the seed of first random object that specified by user (or default).

        Args:
            key(str): Key of item to get.

        Returns:
            int: If key is None return seed for 1st instance (same as what passed to seed function),
                else return seed for specified generator.
        """
        if key is not None:
            return self._seed_dict.get(key, None)

        return self._seed


random = SimRandom()
"""Random utility for simulator, same with original random module."""

seed = random.seed
"""Set seed for simulator."""

__all__ = ['seed', 'random', 'SimRandom']
