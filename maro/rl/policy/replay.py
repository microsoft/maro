# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np


class ReplayMemory:
    """Storage facility for simulation experiences.

    This implementation uses a dictionary of lists as the internal data structure. The objects for each key
    are stored in a list.

    Args:
        capacity (int): Maximum number of experiences that can be stored.
        random_overwrite (bool): This specifies overwrite behavior when the capacity is reached. If this is True,
            overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
            wrap-around. Defaults to False.
    """
    def __init__(self, batch_type, capacity: int, random_overwrite: bool = False):
        super().__init__()
        self._batch_type = batch_type
        self._capacity = capacity
        self._random_overwrite = random_overwrite
        self._keys = batch_type.__slots__
        self.data = {key: [None] * self._capacity for key in self._keys}
        self._size = 0
        self._index = 0

    @property
    def capacity(self):
        """Capacity of the memory."""
        return self._capacity

    @property
    def random_overwrite(self):
        """Overwrite method after the memory has reached capacity."""
        return self._random_overwrite

    @property
    def size(self):
        """Current number of experiences stored."""
        return self._size

    @property
    def keys(self):
        """Keys as specified by ``batch_type``."""
        return self._keys

    def put(self, batch):
        """Put a experience set in the store.
        Args:
            experience_set (ExperienceSet): Experience set to be put in the store.
        """
        assert isinstance(batch, self._batch_type)
        added_size = batch.size
        if added_size > self._capacity:
            raise ValueError("size of added items should not exceed the capacity.")

        num_experiences = self._size + added_size
        num_overwrites = num_experiences - self._capacity
        if num_overwrites <= 0:
            indexes = list(range(self._size, num_experiences))
        # follow the overwrite rule set at init
        elif self._random_overwrite:
            random_indexes = np.random.choice(self._size, size=num_overwrites, replace=False)
            indexes = list(range(self._size, self._capacity)) + list(random_indexes)
        else:
            # using the negative index convention for convenience
            start_index = self._size - self._capacity
            indexes = list(range(start_index, start_index + added_size))

        for key in self.data:
            for idx, val in zip(indexes, getattr(batch, key)):
                self.data[key][idx] = val

        self._size = min(self._capacity, num_experiences)
        return indexes

    def clear(self):
        """Empty the memory."""
        self.data = {key: [None] * self._capacity for key in self._keys}
        self._size = 0
