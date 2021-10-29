# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np


class ReplayMemory:
    """Storage facility for simulation experiences.

    This implementation uses a dictionary of lists as the internal data structure. The objects for each key
    are stored in a list.

    Args:
        capacity (int): Maximum number of experiences that can be stored.
        state_dim (int): Dimension of flattened state.
        action_dim (int): Action dimension. Defaults to 1.
        random_overwrite (bool): This specifies overwrite behavior when the capacity is reached. If this is True,
            overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
            wrap-around. Defaults to False.
    """
    def __init__(self, capacity: int, state_dim: int, action_dim: int = 1, random_overwrite: bool = False):
        super().__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._capacity = capacity
        self._random_overwrite = random_overwrite
        self.states = np.zeros((self._capacity, self._state_dim), dtype=np.float32)
        if action_dim > 1:
            self.actions = np.zeros((self._capacity, self._action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros(self._capacity, dtype=np.int64)
        self.rewards = np.zeros(self._capacity, dtype=np.float32)
        self.next_states = np.zeros((self._capacity, self._state_dim), dtype=np.float32)
        self.terminals = np.zeros(self._capacity, dtype=np.bool)
        self._ptr = 0

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
        return self._ptr

    def put(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        terminals: np.ndarray
    ):
        """Put SARS and terminal flags in the memory."""
        assert len(states) == len(actions) == len(rewards) == len(next_states) == len(terminals)
        added = len(states)
        if added > self._capacity:
            raise ValueError("size of added items should not exceed the capacity.")

        if self._ptr + added <= self._capacity:
            indexes = np.arange(self._ptr, self._ptr + added)
        # follow the overwrite rule set at init
        else:
            overwrites = self._ptr + added - self._capacity
            indexes = np.concatenate([
                np.arange(self._ptr, self._capacity),
                np.random.choice(self._ptr, size=overwrites, replace=False) if self._random_overwrite
                else np.arange(overwrites)
            ])

        self.states[indexes] = states
        self.actions[indexes] = actions
        self.rewards[indexes] = rewards
        self.next_states[indexes] = next_states

        self._ptr = min(self._ptr + added, self._capacity)
        return indexes

    def sample(self, size: int) -> dict:
        """Obtain a random sample."""
        indexes = np.random.choice(self._ptr, size=size)
        return {
            "states": self.states[indexes],
            "actions": self.actions[indexes],
            "rewards": self.rewards[indexes],
            "next_states": self.next_states[indexes],
            "terminals": self.terminals[indexes]
        }
