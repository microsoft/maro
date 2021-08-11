# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.utils.exception.rl_toolkit_exception import InvalidExperience


class ExperienceSet:
    """Wrapper for a set of experiences.

    An experience consists of state, action, reward, next state and auxillary information.
    """
    __slots__ = ["states", "actions", "rewards", "next_states", "info"]

    def __init__(
        self,
        states: list = None,
        actions: list = None,
        rewards: list = None,
        next_states: list = None,
        info: list = None
    ):
        if states is None:
            states, actions, rewards, next_states, info = [], [], [], [], []

        if not len(states) == len(actions) == len(rewards) == len(next_states) == len(info):
            raise InvalidExperience("values of contents should consist of lists of the same length")
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.info = info

    @property
    def size(self):
        return len(self.states)

    def extend(self, other):
        """Concatenate the set with another experience set."""
        self.states += other.states
        self.actions += other.actions
        self.rewards += other.rewards
        self.next_states += other.next_states
        self.info += other.info


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
    def __init__(self, capacity: int, random_overwrite: bool = False):
        super().__init__()
        self._capacity = capacity
        self._random_overwrite = random_overwrite
        self._keys = ExperienceSet.__slots__
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
        """Keys as specified by ``ExperienceSet``."""
        return self._keys

    def put(self, experience_set: ExperienceSet):
        """Put a experience set in the store.
        Args:
            experience_set (ExperienceSet): Experience set to be put in the store.
        """
        added_size = experience_set.size
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
            for idx, val in zip(indexes, getattr(experience_set, key)):
                self.data[key][idx] = val

        self._size = min(self._capacity, num_experiences)
        return indexes

    def clear(self):
        """Empty the memory."""
        self.data = {key: [None] * self._capacity for key in self._keys}
        self._size = 0
