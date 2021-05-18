# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

import numpy as np

from maro.rl.experience import ExperienceSet


class AbsReplayBuffer(ABC):
    """Replay buffer to be used in an EnvWrapper for caching transitions and generating experience sets. 
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def push(self, state, action, reward):
        """Add a new transition to the buffer."""
        raise NotImplementedError

    @abstractmethod
    def batch(self) -> ExperienceSet:
        """Generate an ExperienceSet from the buffer."""
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        """Empty the buffer."""
        raise NotImplementedError


class FIFOReplayBuffer(AbsReplayBuffer):
    """A FIFO-based replay buffer that empties itself after an experience set is generated."""
    def __init__(self):
        super().__init__()
        self.states = []
        self.actions = []
        self.rewards = []    
        self.next_states = []

    def push(self, state, action, reward):
        """Add a new transition to buffer.
        
        If this is not the first transition, the state will correspond to the "next_state" for the transition
        that came before it. 
        """
        if self.states:
            self.next_states.append(state)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def batch(self) -> ExperienceSet:
        """Convert all "SARS" transitions to experience sets and subsequently empty the buffer.

        After this operation, "states", "actions" and "rewards" will have one element remaining that corresponds
        to the last transition for which the next state has yet to be determined. The "next_states" will be empty.
        """
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
        """Empty the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()


class FixedSizeReplayBuffer(AbsReplayBuffer):
    """An implementation of the replay buffer that maintains a fixed-size buffer.
    
    Args:
        capacity (int): Capacity of the buffer. Once the the buffer size has reached capacity, newly added
            transitions will replace existing entries starting from the oldest.
        batch_size (int): Size of experience sets generated from the buffer.
    """
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
        """Add a new transition to buffer.
        
        If this is not the first transition, the state will correspond to the "next_state" for the transition
        that came before it. 
        """
        if self.states[self._index - 1]:
            self.next_states[self._index - 1] = state
            self._size = min(self._size + 1, self.capacity)

        self.states[self._index] = state
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self._index = (self._index + 1) % self.capacity

    def batch(self) -> ExperienceSet:
        """Generate an ExperienceSet from a random sample of transitions in the buffer."""
        indexes = np.random.choice(self._size, size=self.batch_size)
        return ExperienceSet(
            states=list(self.states[indexes]),
            actions=list(self.actions[indexes]),
            rewards=list(self.rewards[indexes]),
            next_states=list(self.next_states[indexes])
        )

    def clear(self):
        """Empty the buffer."""
        self.states = np.empty(self.capacity, dtype=object)
        self.actions = np.empty(self.capacity, dtype=object)
        self.rewards = np.empty(self.capacity, dtype=object)
        self.next_states = np.empty(self.capacity, dtype=object)
        self._size = 0
        self._index = 0
