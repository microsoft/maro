# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

import numpy as np

from .experience import ExperienceSet


class AbsExperienceManager(ABC):
    """Experience memory that stores RL experiences in the form of "state", "action", "reward", "next_state".

    This implementation uses a dictionary of lists as the internal data structure. The objects for each key
    are stored in a list. To be useful for experience storage in RL, uniformity checks are performed during
    put operations to ensure that the list lengths stay the same for all keys at all times. Both unlimited
    and limited storage are supported.

    Args:
        capacity (int): Maximum number of experiences that can be stored.
        overwrite_type (str): If storage capacity is bounded, this specifies how existing entries
            are overwritten when the capacity is exceeded. Two types of overwrite behavior are supported:
            - "rolling", where overwrite occurs sequentially with wrap-around.
            - "random", where overwrite occurs randomly among filled positions.
            Alternatively, the user may also specify overwrite positions (see ``put``).
    """
    def __init__(self, capacity: int, overwrite_type: str = "rolling"):
        super().__init__()
        if overwrite_type not in {"rolling", "random"}:
            raise ValueError(f"overwrite_type must be 'rolling' or 'random', got {overwrite_type}")
        self._capacity = capacity
        self._overwrite_type = overwrite_type
        self.keys = ExperienceSet.__slots__
        self.data = {key: [None] * self._capacity for key in self.keys}
        self._size = 0

    @property
    def size(self):
        return self._size

    @property
    def capacity(self):
        return self._capacity

    @property
    def overwrite_type(self):
        return self._overwrite_type

    @abstractmethod
    def get(self) -> ExperienceSet:
        raise NotImplementedError

    def put(self, experience_set: ExperienceSet):
        """Put new contents in the store.

        Args:
            contents (dict): Dictionary of items to add to the store. If the store is not empty, this must have the
                same keys as the store itself. Otherwise an ``StoreMisalignment`` will be raised.

        Returns:
            The indexes where the newly added entries reside in the store.
        """
        added_size = len(experience_set)
        if added_size > self._capacity:
            raise ValueError("size of added items should not exceed the capacity.")

        num_experiences = self._size + added_size
        num_overwrites = num_experiences - self._capacity
        if num_overwrites <= 0:
            indexes = list(range(self._size, num_experiences))
        # follow the overwrite rule set at init
        elif self._overwrite_type == "rolling":
            # using the negative index convention for convenience
            start_index = self._size - self._capacity
            indexes = list(range(start_index, start_index + added_size))
        else:
            random_indexes = np.random.choice(self._size, size=num_overwrites, replace=False)
            indexes = list(range(self._size, self._capacity)) + list(random_indexes)

        for key in self.data:
            for idx, val in zip(indexes, getattr(experience_set, key)):
                self.data[key][idx] = val

        self._size = min(self._capacity, num_experiences)

    def clear(self):
        """Empty the store."""
        self.data = {key: [None] * self._capacity for key in self.keys}
        self._size = 0


class UniformSampler(AbsExperienceManager):
    """Experience memory that stores RL experiences in the form of "state", "action", "reward", "next_state".

    This implementation uses a dictionary of lists as the internal data structure. The objects for each key
    are stored in a list. To be useful for experience storage in RL, uniformity checks are performed during
    put operations to ensure that the list lengths stay the same for all keys at all times. Both unlimited
    and limited storage are supported.

    Args:
        capacity (int): If negative, the store is of unlimited capacity. Defaults to -1.
        overwrite_type (str): If storage capacity is bounded, this specifies how existing entries
            are overwritten when the capacity is exceeded. Two types of overwrite behavior are supported:
            - "rolling", where overwrite occurs sequentially with wrap-around.
            - "random", where overwrite occurs randomly among filled positions.
            Alternatively, the user may also specify overwrite positions (see ``put``).
    """
    def __init__(self, capacity: int, batch_size: int, overwrite_type: str = None, replace: bool = True):
        super().__init__(capacity, overwrite_type=overwrite_type)
        self.batch_size = batch_size
        self.replace = replace

    def get(self) -> ExperienceSet:
        indexes = np.random.choice(self._size, size=self.batch_size, replace=self.replace)
        return ExperienceSet(*[[self.data[key][idx] for idx in indexes] for key in self.keys])


class UseAndDispose(AbsExperienceManager):
    def get(self) -> ExperienceSet:
        exp_set = ExperienceSet(*[self.data[key] for key in self.keys])
        self.clear()
        return exp_set
