# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

import numpy as np

from maro.rl.experience import ExperienceSet


class AbsReplayBuffer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def push(self, state, action, reward):
        raise NotImplementedError

    @abstractmethod
    def batch(self) -> ExperienceSet:
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        raise NotImplementedError


class FIFOReplayBuffer(AbsReplayBuffer):
    def __init__(self):
        super().__init__()
        self.states = []
        self.actions = []
        self.rewards = []    
        self.next_states = []

    def push(self, state, action, reward):
        if self.states:
            self.next_states.append(state)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def batch(self) -> ExperienceSet:
        exp_set = ExperienceSet(
            states=self.states[:-1],
            actions=self.actions[:-1],
            rewards=self.rewards[:-1],
            next_states=self.next_states
        )

        del self.states[:-1]
        del self.actions[:-1]
        del self.rewards[:-1]
        self.next_states.clear()

        return exp_set

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()


class FixedSizeReplayBuffer(AbsReplayBuffer):
    def __init__(self, capacity: int, batch_size: int):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer since batch_mode is set to 'random'")
        if batch_size > capacity:
            raise ValueError(f"batch_size cannot exceed the buffer capacity ({capacity})")

        self.capacity = capacity
        self.batch_size = batch_size
        self.states = np.empty(capacity, dtype=object)
        self.actions = np.empty(capacity, dtype=object)
        self.rewards = np.empty(capacity, dtype=object)
        self.next_states = np.empty(capacity, dtype=object)
        self._size = 0
        self._index = 0

    def push(self, state, action, reward):
        if self.states[self._index - 1]:
            self.next_states[self._index - 1] = state
            self._size = min(self._size + 1, self.capacity)

        self.states[self._index] = state
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self._index = (self._index + 1) % self.capacity

    def batch(self) -> ExperienceSet:
        indexes = np.random.choice(self._size, size=self.batch_size)
        return ExperienceSet(
            states=list(self.states[indexes]),
            actions=list(self.actions[indexes]),
            rewards=list(self.rewards[indexes]),
            next_states=list(self.next_states[indexes])
        )

    def clear(self):
        self.states = np.empty(self.capacity, dtype=object)
        self.actions = np.empty(self.capacity, dtype=object)
        self.rewards = np.empty(self.capacity, dtype=object)
        self.next_states = np.empty(self.capacity, dtype=object)
        self._size = 0
        self._index = 0
