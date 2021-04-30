# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import Callable, Dict, List, Tuple, Union

import numpy as np

from maro.utils import clone
from maro.utils.exception.rl_toolkit_exception import InvalidExperience


class ExperienceSet:

    __slots__ = ["states", "actions", "rewards", "next_states"]

    def __init__(self, states: list, actions: list, rewards: list, next_states: list):
        if not len(states) == len(actions) == len(rewards) == len(next_states):
            raise InvalidExperience("values of contents should consist of lists of the same length")

        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states

    def __len__(self):
        return len(self.states)


class Replay(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def to_experience_set(self):
        # print(len(self.rewards), len(self.states))
        num_complete = min(len(self.rewards), len(self.states) - 1)
        exp_set = ExperienceSet(
            self.states[:num_complete],
            self.actions[:num_complete],
            self.rewards[:num_complete],
            self.states[1:num_complete + 1]
        )

        del self.states[:num_complete]
        del self.actions[:num_complete]
        del self.rewards[:num_complete]

        return exp_set


class ExperienceMemory(object):
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
    def __init__(self, capacity: int = -1, overwrite_type: str = None):
        super().__init__()
        if overwrite_type not in {"rolling", "random"}:
            raise ValueError(f"overwrite_type must be 'rolling' or 'random', got {overwrite_type}")
        self._capacity = capacity
        self._overwrite_type = overwrite_type
        self._keys = ExperienceSet.__slots__
        self.data = {key: [] if self._capacity == -1 else [None] * self._capacity for key in self._keys}
        self._size = 0

    def __len__(self):
        return self._size

    def __getitem__(self, index: int):
        return {k: lst[index] for k, lst in self.data.items()}

    @property
    def capacity(self):
        """Store capacity.

        If negative, the store grows without bound. Otherwise, the number of items in the store will not exceed
        this capacity.
        """
        return self._capacity

    @property
    def overwrite_type(self):
        """An string indicating the overwrite behavior when the store capacity is exceeded."""
        return self._overwrite_type

    def get(self, indexes: [int] = None) -> ExperienceSet:
        if indexes is None:
            return ExperienceSet(*[self.data[k] for k in self._keys])

        return ExperienceSet(*[[self.data[k][i] for i in indexes] for k in self._keys])

    def put(self, experience_set: ExperienceSet, overwrite_indexes: list = None) -> List[int]:
        """Put new contents in the store.

        Args:
            contents (dict): Dictionary of items to add to the store. If the store is not empty, this must have the
                same keys as the store itself. Otherwise an ``StoreMisalignment`` will be raised.
            overwrite_indexes (list, optional): Indexes where the contents are to be overwritten. This is only
                used when the store has a fixed capacity and putting ``contents`` in the store would exceed this
                capacity. If this is None and overwriting is necessary, rolling or random overwriting will be done
                according to the ``overwrite`` property. Defaults to None.
        Returns:
            The indexes where the newly added entries reside in the store.
        """
        added_size = len(experience_set)
        if self._capacity == -1:
            for key in self.data:
                self.data[key].extend(getattr(experience_set, key))
            self._size += added_size
            return list(range(self._size - added_size, self._size))
        else:
            write_indexes = self._get_update_indexes(added_size, overwrite_indexes=overwrite_indexes)
            for key in self.data:
                for index, value in zip(write_indexes, getattr(experience_set, key)):
                    self.data[key][index] = value

            self._size = min(self._capacity, self._size + added_size)
            return write_indexes

    def clear(self):
        """Empty the store."""
        self.data = {
            key: [] if self._capacity == -1 else [None] * self._capacity for key in ExperienceSet.__slots__
        }
        self._size = 0

    def dumps(self):
        """Return a deep copy of store contents."""
        return clone(dict(self.data))

    def get_by_key(self, key):
        """Get the contents of the store corresponding to ``key``."""
        return self.data[key]

    def _get_update_indexes(self, added_size: int, overwrite_indexes=None):
        if added_size > self._capacity:
            raise ValueError("size of added items should not exceed the store capacity.")

        num_overwrites = self._size + added_size - self._capacity
        if num_overwrites < 0:
            return list(range(self._size, self._size + added_size))

        if overwrite_indexes is not None:
            write_indexes = list(range(self._size, self._capacity)) + list(overwrite_indexes)
        else:
            # follow the overwrite rule set at init
            if self._overwrite_type == "rolling":
                # using the negative index convention for convenience
                start_index = self._size - self._capacity
                write_indexes = list(range(start_index, start_index + added_size))
            else:
                random_indexes = np.random.choice(self._size, size=num_overwrites, replace=False)
                write_indexes = list(range(self._size, self._capacity)) + list(random_indexes)

        return write_indexes
