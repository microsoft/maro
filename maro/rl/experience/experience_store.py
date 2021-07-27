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

    def __getitem__(self, index):
        """Return a copy of ExperienceSet by index."""
        if isinstance(index, int):
            copy = ExperienceSet(
                states=self.states[index:index + 1],
                actions=self.actions[index:index + 1],
                rewards=self.rewards[index:index + 1],
                next_states=self.next_states[index:index + 1],
                info=self.info[index:index + 1])
            return copy
        elif isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            copy = ExperienceSet(
                states=self.states[start:stop:step],
                actions=self.actions[start:stop:step],
                rewards=self.rewards[start:stop:step],
                next_states=self.next_states[start:stop:step],
                info=self.info[start:stop:step])
            return copy
        else:
            raise NotImplementedError


class ExperienceStore:
    """Storage facility for simulation experiences.

    This implementation uses a dictionary of lists as the internal data structure. The objects for each key
    are stored in a list.

    Args:
        capacity (int): Maximum number of experiences that can be stored.
        overwrite_type (str): If storage capacity is bounded, this specifies how existing entries
            are overwritten when the capacity is exceeded. Two types of overwrite behavior are supported:
            - "rolling", where overwrite occurs sequentially with wrap-around.
            - "random", where overwrite occurs randomly among filled positions.
    """
    def __init__(self, capacity: int, overwrite_type: str = "rolling"):
        if overwrite_type not in {"rolling", "random"}:
            raise ValueError(f"overwrite_type must be 'rolling' or 'random', got {overwrite_type}")

        super().__init__()
        self._capacity = capacity
        self._overwrite_type = overwrite_type
        self._keys = ExperienceSet.__slots__
        self.data = {key: [None] * self._capacity for key in self._keys}
        self._size = 0
        self._index = 0

    @property
    def capacity(self):
        """Capacity of the memory."""
        return self._capacity

    @property
    def overwrite_type(self):
        """Overwrite method after the memory has reached capacity."""
        return self._overwrite_type

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
            experience_set (ExperienceSet): Experience set to be put in the store. If the store is full,
                existing items will be overwritten according to the ``overwrite_type`` property.

        """
        added_size = experience_set.size
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
        """Empty the memory."""
        self.data = {key: [None] * self._capacity for key in self._keys}
        self._size = 0
