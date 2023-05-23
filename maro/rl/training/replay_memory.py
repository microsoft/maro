# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from maro.rl.utils import SHAPE_CHECK_FLAG, MultiTransitionBatch, TransitionBatch, match_shape


class AbsIndexScheduler(object, metaclass=ABCMeta):
    """Scheduling indexes for read and write requests. This is used as an inner module of the replay memory.

    Args:
        capacity (int): Maximum capacity of the replay memory.
    """

    def __init__(self, capacity: int) -> None:
        super(AbsIndexScheduler, self).__init__()
        self._capacity = capacity

    @abstractmethod
    def get_put_indexes(self, batch_size: int) -> np.ndarray:
        """Generate a list of indexes to the replay memory for writing. In other words, when the replay memory
        need to write a batch, the scheduler should provide a set of proper indexes for the replay memory to
        write.

        Args:
            batch_size (int): The required batch size.

        Returns:
            indexes (np.ndarray): The list of indexes.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sample_indexes(self, batch_size: int = None) -> np.ndarray:
        """Generate a list of indexes that can be used to retrieve items from the replay memory.

        Args:
            batch_size (int, default=None): The required batch size. If it is None, all indexes where an experience
                item is present are returned.

        Returns:
            indexes (np.ndarray): The list of indexes.
        """
        raise NotImplementedError


class RandomIndexScheduler(AbsIndexScheduler):
    """Index scheduler that returns random indexes when sampling.

    Args:
        capacity (int): Maximum capacity of the replay memory.
        random_overwrite (bool): Flag that controls the overwriting behavior when the replay memory reaches capacity.
            If this is true, newly added items will randomly overwrite existing ones. Otherwise, the overwrite occurs
            in a cyclic manner.
    """

    def __init__(self, capacity: int, random_overwrite: bool) -> None:
        super(RandomIndexScheduler, self).__init__(capacity)
        self._random_overwrite = random_overwrite
        self._ptr = self._size = 0

    def get_put_indexes(self, batch_size: int) -> np.ndarray:
        if self._ptr + batch_size <= self._capacity:
            indexes = np.arange(self._ptr, self._ptr + batch_size)
            self._ptr += batch_size
        else:
            overwrites = self._ptr + batch_size - self._capacity
            indexes = np.concatenate(
                [
                    np.arange(self._ptr, self._capacity),
                    np.random.choice(self._ptr, size=overwrites, replace=False)
                    if self._random_overwrite
                    else np.arange(overwrites),
                ],
            )
            self._ptr = self._capacity if self._random_overwrite else overwrites

        self._size = min(self._size + batch_size, self._capacity)
        return indexes

    def get_sample_indexes(self, batch_size: int = None) -> np.ndarray:
        assert batch_size is not None and batch_size > 0, f"Invalid batch size: {batch_size}"
        assert self._size > 0, "Cannot sample from an empty memory."
        return np.random.choice(self._size, size=batch_size, replace=True)


class PriorityReplayIndexScheduler(AbsIndexScheduler):
    """
    Indexer for priority replay memory: https://arxiv.org/abs/1511.05952.

    Args:
        capacity (int): Maximum capacity of the replay memory.
        alpha (float): Alpha (see original paper for explanation).
        beta (float): Alpha (see original paper for explanation).
    """

    def __init__(
        self,
        capacity: int,
        alpha: float,
        beta: float,
    ) -> None:
        super(PriorityReplayIndexScheduler, self).__init__(capacity)
        self._alpha = alpha
        self._beta = beta
        self._max_prio = self._min_prio = 1.0
        self._weights = np.zeros(capacity, dtype=np.float32)

        self._ptr = self._size = 0

    def init_weights(self, indexes: np.ndarray) -> None:
        self._weights[indexes] = self._max_prio**self._alpha

    def get_weight(self, indexes: np.ndarray) -> np.ndarray:
        # important sampling weight calculation
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        return (self._weights[indexes] / self._min_prio) ** (-self._beta)

    def update_weight(self, indexes: np.ndarray, weight: np.ndarray) -> None:
        assert indexes.shape == weight.shape
        weight = np.abs(weight) + np.finfo(np.float32).eps.item()
        self._weights[indexes] = weight**self._alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

    def get_put_indexes(self, batch_size: int) -> np.ndarray:
        if self._ptr + batch_size <= self._capacity:
            indexes = np.arange(self._ptr, self._ptr + batch_size)
            self._ptr += batch_size
        else:
            overwrites = self._ptr + batch_size - self._capacity
            indexes = np.concatenate(
                [
                    np.arange(self._ptr, self._capacity),
                    np.arange(overwrites),
                ],
            )
            self._ptr = overwrites

        self._size = min(self._size + batch_size, self._capacity)
        self.init_weights(indexes)
        return indexes

    def get_sample_indexes(self, batch_size: int = None) -> np.ndarray:
        assert batch_size is not None and batch_size > 0, f"Invalid batch size: {batch_size}"
        assert self._size > 0, "Cannot sample from an empty memory."

        weights = self._weights[: self._size]
        weights = weights / weights.sum()
        return np.random.choice(np.arange(self._size), p=weights, size=batch_size, replace=True)


class FIFOIndexScheduler(AbsIndexScheduler):
    """First-in-first-out index scheduler.

    Args:
        capacity (int): Maximum capacity of the replay memory.
    """

    def __init__(self, capacity: int) -> None:
        super(FIFOIndexScheduler, self).__init__(capacity)
        self._head = self._tail = 0

    @property
    def size(self) -> int:
        return (self._tail - self._head) % self._capacity

    def get_put_indexes(self, batch_size: int) -> np.ndarray:
        if self.size + batch_size <= self._capacity:
            if self._tail + batch_size <= self._capacity:
                indexes = np.arange(self._tail, self._tail + batch_size)
            else:
                indexes = np.concatenate(
                    [
                        np.arange(self._tail, self._capacity),
                        np.arange(self._tail + batch_size - self._capacity),
                    ],
                )
            self._tail = (self._tail + batch_size) % self._capacity
            return indexes
        else:
            overwrite = self.size + batch_size - self._capacity
            self._head = (self._head + overwrite) % self._capacity
            return self.get_put_indexes(batch_size)

    def get_sample_indexes(self, batch_size: int = None) -> np.ndarray:
        indexes = (
            np.arange(self._head, self._tail)
            if self._tail > self._head
            else np.concatenate([np.arange(self._head, self._capacity), np.arange(self._tail)])
        )
        self._head = self._tail
        return indexes


class AbsReplayMemory(object, metaclass=ABCMeta):
    """Abstract replay memory class with basic interfaces.

    Args:
        capacity (int): Maximum capacity of the replay memory.
        state_dim (int): Dimension of states.
        idx_scheduler (AbsIndexScheduler): The index scheduler.
    """

    def __init__(self, capacity: int, state_dim: int, idx_scheduler: AbsIndexScheduler) -> None:
        super(AbsReplayMemory, self).__init__()
        self._capacity = capacity
        self._state_dim = state_dim
        self._idx_scheduler = idx_scheduler

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def get_put_indexes(self, batch_size: int) -> np.ndarray:
        """Please refer to the doc string in AbsIndexScheduler."""
        return self._idx_scheduler.get_put_indexes(batch_size)

    def get_sample_indexes(self, batch_size: int = None) -> np.ndarray:
        """Please refer to the doc string in AbsIndexScheduler."""
        return self._idx_scheduler.get_sample_indexes(batch_size)


class ReplayMemory(AbsReplayMemory, metaclass=ABCMeta):
    """In-memory experience storage facility for a single trainer.

    Args:
        capacity (int): Maximum capacity of the replay memory.
        state_dim (int): Dimension of states.
        action_dim (int): Dimension of actions.
        idx_scheduler (AbsIndexScheduler): The index scheduler.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        idx_scheduler: AbsIndexScheduler,
    ) -> None:
        super(ReplayMemory, self).__init__(capacity, state_dim, idx_scheduler)
        self._action_dim = action_dim

        self._states = np.zeros((self._capacity, self._state_dim), dtype=np.float32)
        self._actions = np.zeros((self._capacity, self._action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._capacity, dtype=np.float32)
        self._terminals = np.zeros(self._capacity, dtype=bool)
        self._truncated = np.zeros(self._capacity, dtype=bool)
        self._next_states = np.zeros((self._capacity, self._state_dim), dtype=np.float32)
        self._returns = np.zeros(self._capacity, dtype=np.float32)
        self._advantages = np.zeros(self._capacity, dtype=np.float32)
        self._old_logps = np.zeros(self._capacity, dtype=np.float32)

        self._n_sample = 0

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def n_sample(self) -> int:
        return self._n_sample

    def put(self, transition_batch: TransitionBatch) -> None:
        """Store a transition batch in the memory.

        Args:
            transition_batch (TransitionBatch): The transition batch.
        """
        batch_size = len(transition_batch.states)
        if SHAPE_CHECK_FLAG:
            assert 0 < batch_size <= self._capacity
            assert match_shape(transition_batch.states, (batch_size, self._state_dim))
            assert match_shape(transition_batch.actions, (batch_size, self._action_dim))
            assert match_shape(transition_batch.rewards, (batch_size,))
            assert match_shape(transition_batch.terminals, (batch_size,))
            assert match_shape(transition_batch.truncated, (batch_size,))
            assert match_shape(transition_batch.next_states, (batch_size, self._state_dim))
            if transition_batch.returns is not None:
                match_shape(transition_batch.returns, (batch_size,))
            if transition_batch.advantages is not None:
                match_shape(transition_batch.advantages, (batch_size,))
            if transition_batch.old_logps is not None:
                match_shape(transition_batch.old_logps, (batch_size,))

        self.put_by_indexes(self.get_put_indexes(batch_size), transition_batch)
        self._n_sample = min(self._n_sample + transition_batch.size, self._capacity)

    def put_by_indexes(self, indexes: np.ndarray, transition_batch: TransitionBatch) -> None:
        """Store a transition batch into the memory at the give indexes.

        Args:
            indexes (np.ndarray): Positions in the replay memory to store at.
            transition_batch (TransitionBatch): The transition batch.
        """
        self._states[indexes] = transition_batch.states
        self._actions[indexes] = transition_batch.actions
        self._rewards[indexes] = transition_batch.rewards
        self._terminals[indexes] = transition_batch.terminals
        self._truncated[indexes] = transition_batch.truncated
        self._next_states[indexes] = transition_batch.next_states
        if transition_batch.returns is not None:
            self._returns[indexes] = transition_batch.returns
        if transition_batch.advantages is not None:
            self._advantages[indexes] = transition_batch.advantages
        if transition_batch.old_logps is not None:
            self._old_logps[indexes] = transition_batch.old_logps

    def sample(self, batch_size: int = None) -> TransitionBatch:
        """Generate a sample batch from the replay memory.

        Args:
            batch_size (int, default=None): The required batch size. If it is None, all indexes where an experience
                item is present are returned.

        Returns:
            batch (TransitionBatch): The sampled batch.
        """
        indexes = self.get_sample_indexes(batch_size)
        return self.sample_by_indexes(indexes)

    def sample_by_indexes(self, indexes: np.ndarray) -> TransitionBatch:
        """Retrieve items at given indexes from the replay memory.

        Args:
            indexes (np.ndarray): Positions in the replay memory to retrieve at.

        Returns:
            batch (TransitionBatch): The sampled batch.
        """
        assert all([0 <= idx < self._capacity for idx in indexes])

        return TransitionBatch(
            states=self._states[indexes],
            actions=self._actions[indexes],
            rewards=self._rewards[indexes],
            terminals=self._terminals[indexes],
            truncated=self._truncated[indexes],
            next_states=self._next_states[indexes],
            returns=self._returns[indexes],
            advantages=self._advantages[indexes],
            old_logps=self._old_logps[indexes],
        )


class RandomReplayMemory(ReplayMemory):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        random_overwrite: bool = False,
    ) -> None:
        super(RandomReplayMemory, self).__init__(
            capacity,
            state_dim,
            action_dim,
            RandomIndexScheduler(capacity, random_overwrite),
        )
        self._random_overwrite = random_overwrite

    @property
    def random_overwrite(self) -> bool:
        return self._random_overwrite


class PrioritizedReplayMemory(ReplayMemory):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        alpha: float,
        beta: float,
    ) -> None:
        super(PrioritizedReplayMemory, self).__init__(
            capacity,
            state_dim,
            action_dim,
            PriorityReplayIndexScheduler(capacity, alpha, beta),
        )

    def get_weight(self, indexes: np.ndarray) -> np.ndarray:
        assert isinstance(self._idx_scheduler, PriorityReplayIndexScheduler)
        return self._idx_scheduler.get_weight(indexes)

    def update_weight(self, indexes: np.ndarray, weight: np.ndarray) -> None:
        assert isinstance(self._idx_scheduler, PriorityReplayIndexScheduler)
        self._idx_scheduler.update_weight(indexes, weight)


class FIFOReplayMemory(ReplayMemory):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
    ) -> None:
        super(FIFOReplayMemory, self).__init__(
            capacity,
            state_dim,
            action_dim,
            FIFOIndexScheduler(capacity),
        )


class MultiReplayMemory(AbsReplayMemory, metaclass=ABCMeta):
    """In-memory experience storage facility for a multi trainer.

    Args:
        capacity (int): Maximum capacity of the replay memory.
        state_dim (int): Dimension of states.
        action_dims (List[int]): Dimensions of actions.
        idx_scheduler (AbsIndexScheduler): The index scheduler.
        agent_states_dims (List[int]): Dimensions of agent states.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dims: List[int],
        idx_scheduler: AbsIndexScheduler,
        agent_states_dims: List[int],
    ) -> None:
        super(MultiReplayMemory, self).__init__(capacity, state_dim, idx_scheduler)
        self._agent_num = len(action_dims)
        self._action_dims = action_dims

        self._states = np.zeros((self._capacity, self._state_dim), dtype=np.float32)
        self._actions = [np.zeros((self._capacity, action_dim), dtype=np.float32) for action_dim in self._action_dims]
        self._rewards = [np.zeros(self._capacity, dtype=np.float32) for _ in range(self.agent_num)]
        self._next_states = np.zeros((self._capacity, self._state_dim), dtype=np.float32)
        self._terminals = np.zeros(self._capacity, dtype=bool)
        self._truncated = np.zeros(self._capacity, dtype=bool)

        assert len(agent_states_dims) == self.agent_num
        self._agent_states_dims = agent_states_dims
        self._agent_states = [
            np.zeros((self._capacity, state_dim), dtype=np.float32) for state_dim in self._agent_states_dims
        ]
        self._next_agent_states = [
            np.zeros((self._capacity, state_dim), dtype=np.float32) for state_dim in self._agent_states_dims
        ]

    @property
    def action_dims(self) -> List[int]:
        return self._action_dims

    @property
    def agent_num(self) -> int:
        return self._agent_num

    def put(self, transition_batch: MultiTransitionBatch) -> None:
        """Store a transition batch into the memory.

        Args:
            transition_batch (MultiTransitionBatch): The transition batch.
        """
        batch_size = len(transition_batch.states)
        if SHAPE_CHECK_FLAG:
            assert 0 < batch_size <= self._capacity
            assert match_shape(transition_batch.states, (batch_size, self._state_dim))
            assert len(transition_batch.actions) == len(transition_batch.rewards) == self.agent_num
            for i in range(self.agent_num):
                assert match_shape(transition_batch.actions[i], (batch_size, self.action_dims[i]))
                assert match_shape(transition_batch.rewards[i], (batch_size,))

            assert match_shape(transition_batch.terminals, (batch_size,))
            assert match_shape(transition_batch.truncated, (batch_size,))
            assert match_shape(transition_batch.next_states, (batch_size, self._state_dim))

            assert len(transition_batch.agent_states) == self.agent_num
            assert len(transition_batch.next_agent_states) == self.agent_num
            for i in range(self.agent_num):
                assert match_shape(transition_batch.agent_states[i], (batch_size, self._agent_states_dims[i]))
                assert match_shape(transition_batch.next_agent_states[i], (batch_size, self._agent_states_dims[i]))

        self.put_by_indexes(self.get_put_indexes(batch_size), transition_batch=transition_batch)

    def put_by_indexes(self, indexes: np.ndarray, transition_batch: MultiTransitionBatch) -> None:
        """Store a transition batch into the memory at the give indexes.

        Args:
            indexes (np.ndarray): Positions in the replay memory to store at.
            transition_batch (MultiTransitionBatch): The transition batch.
        """
        self._states[indexes] = transition_batch.states
        for i in range(self.agent_num):
            self._actions[i][indexes] = transition_batch.actions[i]
            self._rewards[i][indexes] = transition_batch.rewards[i]
        self._terminals[indexes] = transition_batch.terminals
        self._truncated[indexes] = transition_batch.truncated

        self._next_states[indexes] = transition_batch.next_states
        for i in range(self.agent_num):
            self._agent_states[i][indexes] = transition_batch.agent_states[i]
            self._next_agent_states[i][indexes] = transition_batch.next_agent_states[i]

    def sample(self, batch_size: int = None) -> MultiTransitionBatch:
        """Generate a sample batch from the replay memory.

        Args:
            batch_size (int, default=None): The required batch size. If it is None, all indexes where an experience
                item is present are returned.

        Returns:
            batch (MultiTransitionBatch): The sampled batch.
        """
        indexes = self.get_sample_indexes(batch_size)
        return self.sample_by_indexes(indexes)

    def sample_by_indexes(self, indexes: np.ndarray) -> MultiTransitionBatch:
        """Retrieve items at given indexes from the replay memory.

        Args:
            indexes (np.ndarray): Positions in the replay memory to retrieve at.

        Returns:
            batch (MultiTransitionBatch): The sampled batch.
        """
        assert all([0 <= idx < self._capacity for idx in indexes])

        return MultiTransitionBatch(
            states=self._states[indexes],
            actions=[action[indexes] for action in self._actions],
            rewards=[reward[indexes] for reward in self._rewards],
            terminals=self._terminals[indexes],
            truncated=self._truncated[indexes],
            next_states=self._next_states[indexes],
            agent_states=[state[indexes] for state in self._agent_states],
            next_agent_states=[state[indexes] for state in self._next_agent_states],
        )


class RandomMultiReplayMemory(MultiReplayMemory):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dims: List[int],
        agent_states_dims: List[int],
        random_overwrite: bool = False,
    ) -> None:
        super(RandomMultiReplayMemory, self).__init__(
            capacity,
            state_dim,
            action_dims,
            RandomIndexScheduler(capacity, random_overwrite),
            agent_states_dims,
        )
        self._random_overwrite = random_overwrite

    @property
    def random_overwrite(self) -> bool:
        return self._random_overwrite


class FIFOMultiReplayMemory(MultiReplayMemory):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dims: List[int],
        agent_states_dims: List[int],
    ) -> None:
        super(FIFOMultiReplayMemory, self).__init__(
            capacity,
            state_dim,
            action_dims,
            FIFOIndexScheduler(capacity),
            agent_states_dims,
        )
