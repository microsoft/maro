from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from maro.rl_v3.utils import MultiTransitionBatch, SHAPE_CHECK_FLAG, TransitionBatch, match_shape


class AbsIndexScheduler(object, metaclass=ABCMeta):
    def __init__(self, capacity: int) -> None:
        super(AbsIndexScheduler, self).__init__()
        self._capacity = capacity

    @abstractmethod
    def get_put_indexes(self, batch_size: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_sample_indexes(self, batch_size: int = None, forbid_last: bool = False) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_last_index(self) -> int:
        raise NotImplementedError


class RandomIndexScheduler(AbsIndexScheduler):
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
            indexes = np.concatenate([
                np.arange(self._ptr, self._capacity),
                np.random.choice(self._ptr, size=overwrites, replace=False) if self._random_overwrite
                else np.arange(overwrites)
            ])
            self._ptr = self._capacity if self._random_overwrite else overwrites

        self._size = min(self._size + batch_size, self._capacity)
        return indexes

    def get_sample_indexes(self, batch_size: int = None, forbid_last: bool = False) -> np.ndarray:
        assert batch_size is not None and batch_size > 0, f"Invalid batch size: {batch_size}"
        assert self._size > 0, "Cannot sample from an empty memory."
        return np.random.choice(self._size, size=batch_size, replace=True)

    def get_last_index(self) -> int:
        raise NotImplementedError


class FIFOIndexScheduler(AbsIndexScheduler):
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
                indexes = np.concatenate([
                    np.arange(self._tail, self._capacity),
                    np.arange(self._tail + batch_size - self._capacity)
                ])
            self._tail = (self._tail + batch_size) % self._capacity
            return indexes
        else:
            overwrite = self.size + batch_size - self._capacity
            self._head = (self._head + overwrite) % self._capacity
            return self.get_put_indexes(batch_size)

    def get_sample_indexes(self, batch_size: int = None, forbid_last: bool = False) -> np.ndarray:
        tmp = self._tail if not forbid_last else (self._tail - 1) % self._capacity
        indexes = np.arange(self._head, tmp) if tmp > self._head \
            else np.concatenate([np.arange(self._head, self._capacity), np.arange(tmp)])
        self._head = tmp
        return indexes

    def get_last_index(self) -> int:
        return (self._tail - 1) % self._capacity


class AbsReplayMemory(object, metaclass=ABCMeta):
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

    def _get_put_indexes(self, batch_size: int) -> np.ndarray:
        return self._idx_scheduler.get_put_indexes(batch_size)

    def _get_sample_indexes(self, batch_size: int = None, forbid_last: bool = False) -> np.ndarray:
        return self._idx_scheduler.get_sample_indexes(batch_size, forbid_last)


class ReplayMemory(AbsReplayMemory, metaclass=ABCMeta):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        idx_scheduler: AbsIndexScheduler,
        enable_next_states: bool = True,
        enable_values: bool = True,
        enable_logps: bool = True
    ) -> None:
        super(ReplayMemory, self).__init__(capacity, state_dim, idx_scheduler)
        self._action_dim = action_dim

        self._states = np.zeros((self._capacity, self._state_dim), dtype=np.float32)
        self._actions = np.zeros((self._capacity, self._action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._capacity, dtype=np.float32)
        self._terminals = np.zeros(self._capacity, dtype=np.bool)

        self._enable_next_states = enable_next_states
        self._enable_values = enable_values
        self._enable_logps = enable_logps

        self._next_states = None if not self._enable_next_states \
            else np.zeros((self._capacity, self._state_dim), dtype=np.float32)
        self._values = None if not self._enable_values else np.zeros(self._capacity, dtype=np.float32)
        self._logps = None if not self._enable_logps else np.zeros(self._capacity, dtype=np.float32)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def put(self, transition_batch: TransitionBatch) -> None:
        batch_size = len(transition_batch.states)
        if SHAPE_CHECK_FLAG:
            assert 0 < batch_size <= self._capacity
            assert match_shape(transition_batch.states, (batch_size, self._state_dim))
            assert match_shape(transition_batch.actions, (batch_size, self._action_dim))
            assert match_shape(transition_batch.rewards, (batch_size,))
            assert match_shape(transition_batch.terminals, (batch_size,))
            if transition_batch.next_states is not None:
                assert match_shape(transition_batch.next_states, (batch_size, self._state_dim))
            if transition_batch.values is not None:
                assert match_shape(transition_batch.values, (batch_size,))
            if transition_batch.logps is not None:
                assert match_shape(transition_batch.logps, (batch_size,))

        self._put_by_indexes(self._get_put_indexes(batch_size), transition_batch)

    def _put_by_indexes(self, indexes: np.ndarray, transition_batch: TransitionBatch):
        self._states[indexes] = transition_batch.states
        self._actions[indexes] = transition_batch.actions
        self._rewards[indexes] = transition_batch.rewards
        self._terminals[indexes] = transition_batch.terminals
        if transition_batch.next_states is not None:
            self._next_states[indexes] = transition_batch.next_states
        if transition_batch.values is not None:
            self._values[indexes] = transition_batch.values
        if transition_batch.logps is not None:
            self._logps[indexes] = transition_batch.logps

    def sample(self, batch_size: int = None) -> TransitionBatch:
        indexes = self._get_sample_indexes(batch_size, self._get_forbid_last())
        return self.sample_by_indexes(indexes)

    def sample_by_indexes(self, indexes: np.ndarray) -> TransitionBatch:
        assert all([0 <= idx < self._capacity for idx in indexes])

        return TransitionBatch(
            policy_name='',
            states=self._states[indexes],
            actions=self._actions[indexes],
            rewards=self._rewards[indexes],
            terminals=self._terminals[indexes],
            next_states=self._next_states[indexes] if self._enable_next_states else None,
            values=self._values[indexes] if self._enable_values else None,
            logps=self._logps[indexes] if self._enable_logps else None
        )

    @abstractmethod
    def _get_forbid_last(self) -> bool:
        raise NotImplementedError


class RandomReplayMemory(ReplayMemory):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        random_overwrite: bool = False,
        enable_next_states: bool = True,
        enable_values: bool = True,
        enable_logps: bool = True
    ):
        super(RandomReplayMemory, self).__init__(
            capacity, state_dim, action_dim, RandomIndexScheduler(capacity, random_overwrite),
            enable_next_states, enable_values, enable_logps
        )
        self._random_overwrite = random_overwrite
        self._scheduler = RandomIndexScheduler(capacity, random_overwrite)

    @property
    def random_overwrite(self):
        return self._random_overwrite

    def _get_forbid_last(self) -> bool:
        return False


class FIFOReplayMemory(ReplayMemory):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        enable_next_states: bool = True,
        enable_values: bool = True,
        enable_logps: bool = True
    ):
        super(FIFOReplayMemory, self).__init__(
            capacity, state_dim, action_dim, FIFOIndexScheduler(capacity),
            enable_next_states, enable_values, enable_logps
        )

    def _get_forbid_last(self) -> bool:
        return not self._terminals[self._idx_scheduler.get_last_index()]


class MultiReplayMemory(AbsReplayMemory, metaclass=ABCMeta):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dims: List[int],
        idx_scheduler: AbsIndexScheduler,
        enable_local_states: bool = True,
        local_states_dims: List[int] = None,
        enable_next_states: bool = True,
        enable_values: bool = True,
        enable_logps: bool = True
    ) -> None:
        super(MultiReplayMemory, self).__init__(capacity, state_dim, idx_scheduler)
        self._agent_num = len(action_dims)
        self._action_dims = action_dims

        self._states = np.zeros((self._capacity, self._state_dim), dtype=np.float32)
        self._actions = [np.zeros((self._capacity, action_dim), dtype=np.float32) for action_dim in self._action_dims]
        self._rewards = [np.zeros(self._capacity, dtype=np.float32) for _ in range(self.agent_num)]
        self._terminals = np.zeros(self._capacity, dtype=np.bool)

        self._enable_local_states = enable_local_states
        self._enable_next_states = enable_next_states
        self._enable_values = enable_values
        self._enable_logps = enable_logps

        if self._enable_local_states:
            assert local_states_dims is not None and len(local_states_dims) == self.agent_num
            self._local_states_dims = local_states_dims
            self._local_states = [
                np.zeros((self._capacity, state_dim), dtype=np.float32) for state_dim in self._local_states_dims
            ]

        self._next_states = None if not self._enable_next_states \
            else np.zeros((self._capacity, self._state_dim), dtype=np.float32)
        self._values = None if not self._enable_values else np.zeros(self._capacity, dtype=np.float32)
        self._logps = None if not self._enable_logps else np.zeros(self._capacity, dtype=np.float32)

    @property
    def action_dims(self) -> List[int]:
        return self._action_dims

    @property
    def agent_num(self) -> int:
        return self._agent_num

    def put(self, transition_batch: MultiTransitionBatch) -> None:
        batch_size = len(transition_batch.states)
        if SHAPE_CHECK_FLAG:
            assert 0 < batch_size <= self._capacity
            assert match_shape(transition_batch.states, (batch_size, self._state_dim))
            assert len(transition_batch.actions) == len(transition_batch.rewards) == self.agent_num
            for i in range(self.agent_num):
                assert match_shape(transition_batch.actions[i], (batch_size, self.action_dims[i]))
                assert match_shape(transition_batch.rewards[i], (batch_size,))

            assert match_shape(transition_batch.terminals, (batch_size,))

            if self._enable_local_states:
                assert transition_batch.agent_states is not None
                assert len(transition_batch.agent_states) == self.agent_num
                for i in range(self.agent_num):
                    assert match_shape(transition_batch.agent_states[i], (batch_size, self._local_states_dims[i]))

            if transition_batch.next_states is not None:
                assert match_shape(transition_batch.next_states, (batch_size, self._state_dim))
            if transition_batch.values is not None:
                for value in transition_batch.values:
                    assert match_shape(value, (batch_size,))
            if transition_batch.logps is not None:
                for logp in transition_batch.logps:
                    assert match_shape(logp, (batch_size,))

        self._put_by_indexes(self._get_put_indexes(batch_size), transition_batch=transition_batch)

    def _put_by_indexes(self, indexes: np.ndarray, transition_batch: MultiTransitionBatch):
        self._states[indexes] = transition_batch.states
        for i in range(self.agent_num):
            self._actions[i][indexes] = transition_batch.actions[i]
            self._rewards[i][indexes] = transition_batch.rewards[i]
        self._terminals[indexes] = transition_batch.terminals
        if transition_batch.agent_states is not None:
            for i in range(self.agent_num):
                self._local_states[i][indexes] = transition_batch.agent_states[i]
        if transition_batch.next_states is not None:
            self._next_states[indexes] = transition_batch.next_states
        if transition_batch.values is not None:
            self._values[indexes] = transition_batch.values
        if transition_batch.logps is not None:
            self._logps[indexes] = transition_batch.logps

    def sample(self, batch_size: int = None) -> MultiTransitionBatch:
        indexes = self._get_sample_indexes(batch_size, self._get_forbid_last())
        return self.sample_by_indexes(indexes)

    def sample_by_indexes(self, indexes: np.ndarray) -> MultiTransitionBatch:
        assert all([0 <= idx < self._capacity for idx in indexes])

        return MultiTransitionBatch(
            policy_names=[],
            states=self._states[indexes],
            actions=[action[indexes] for action in self._actions],
            rewards=[reward[indexes] for reward in self._rewards],
            terminals=self._terminals[indexes],
            agent_states=[state[indexes] for state in self._local_states] if self._enable_local_states else None,
            next_states=self._next_states[indexes] if self._enable_next_states else None,
            values=self._values[indexes] if self._enable_values else None,
            logps=self._logps[indexes] if self._enable_logps else None
        )

    @abstractmethod
    def _get_forbid_last(self) -> bool:
        raise NotImplementedError


class RandomMultiReplayMemory(MultiReplayMemory):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dims: List[int],
        random_overwrite: bool = False,
        enable_local_states: bool = True,
        local_states_dims: List[int] = None,
        enable_next_states: bool = True,
        enable_values: bool = True,
        enable_logps: bool = True
    ):
        super(RandomMultiReplayMemory, self).__init__(
            capacity, state_dim, action_dims, RandomIndexScheduler(capacity, random_overwrite),
            enable_local_states, local_states_dims, enable_next_states, enable_values, enable_logps
        )
        self._random_overwrite = random_overwrite
        self._scheduler = RandomIndexScheduler(capacity, random_overwrite)

    @property
    def random_overwrite(self):
        return self._random_overwrite

    def _get_forbid_last(self) -> bool:
        return False


class FIFOMultiReplayMemory(MultiReplayMemory):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dims: List[int],
        enable_local_states: bool = True,
        local_states_dims: List[int] = None,
        enable_next_states: bool = True,
        enable_values: bool = True,
        enable_logps: bool = True
    ):
        super(FIFOMultiReplayMemory, self).__init__(
            capacity, state_dim, action_dims, FIFOIndexScheduler(capacity),
            enable_local_states, local_states_dims, enable_next_states, enable_values, enable_logps
        )

    def _get_forbid_last(self) -> bool:
        return not self._terminals[self._idx_scheduler.get_last_index()]


# if __name__ == '__main__':
    # memory = FIFOReplayMemory(capacity=10, state_dim=5, action_dim=3)
    # for i in range(1, 10):
    #     print(f"\nput i")
    #     memory.put(
    #         states=np.ones((i, 5)),
    #         actions=np.ones((i, 3)),
    #         rewards=np.ones(i),
    #         terminals=np.zeros(i),
    #         next_states=np.ones((i, 5)),
    #         values=np.ones(i),
    #         logps=np.ones(i)
    #     )
    #
    #     print(memory._idx_scheduler._head, memory._idx_scheduler._tail, memory._idx_scheduler._size)
    #     if i % 3 == 0:
    #         memory.sample()
    #         print(memory._idx_scheduler._head, memory._idx_scheduler._tail, memory._idx_scheduler._size)
    #
    # memory = RandomReplayMemory(capacity=10, state_dim=5, action_dim=3, random_overwrite=True)
    # for i in range(1, 10):
    #     print(f"\nput i")
    #     memory.put(
    #         states=np.ones((i, 5)),
    #         actions=np.ones((i, 3)),
    #         rewards=np.ones(i),
    #         terminals=np.zeros(i),
    #         next_states=np.ones((i, 5)),
    #         values=np.ones(i),
    #         logps=np.ones(i)
    #     )
    #
    #     print(memory._idx_scheduler._size, memory._idx_scheduler._ptr)
    #     if i % 3 == 0:
    #         memory.sample(3)
    #         print(memory._idx_scheduler._size, memory._idx_scheduler._ptr)
