# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import warnings
from typing import Dict, List, Optional

import numpy as np
from tianshou.data import Batch
from tianshou.data.batch import _alloc_by_keys_diff, _create_value

REQUIRED_KEYS = {"obs", "action", "reward", "next_obs", "terminal", "truncated"}
SUPPORTED_KEYS = REQUIRED_KEYS  # TODO: check if this is needed


class ReplayMemory(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data = Batch()
        self.size = self.ptr = 0

    def reset(self) -> None:
        self.data = Batch()
        self.size = self.ptr = 0

    @property
    def header(self) -> int:
        return (self.ptr - self.size) % self.capacity

    def _get_contiguous_indexes(self, start: int, size: int) -> np.ndarray:
        if start + size < self.capacity:
            return np.arange(start, start + size)
        else:
            return np.concatenate(
                [
                    np.arange(start, self.capacity),
                    np.arange(0, start + size - self.capacity),
                ],
            )

    def sample(self, size: Optional[int] = None, random: bool = False, pop: bool = False) -> Batch:
        if size is None:
            indexes = self._get_contiguous_indexes(self.header, self.size)
            batch = self.data[indexes]
            if pop:
                self.reset()
            return batch
        else:
            assert 0 < size <= self.size

            if random:
                assert not pop, "Pop is not allowed under random mode."
                indexes = np.random.choice(np.arange(self.size), size=size, replace=False)
                return self.data[indexes]
            else:
                t = self.header
                indexes = self._get_contiguous_indexes(t, size)
                batch = self.data[indexes]
                if pop:
                    self.size -= size  # Reduce self_size only. Pop elements will not affect self._ptr.
                return batch

    def store(self, batch: Batch) -> None:
        assert REQUIRED_KEYS.issubset(batch.keys())

        if len(batch) > self.capacity:
            warnings.warn(
                f"Trying to store a HUGE batch of size {len(batch)} into a replay memory of size {self.capacity}. "
                f"Only the last {self.capacity} elements will be kept.",
            )
            batch = batch[-self.capacity :]

        n = len(batch)
        indexes = self._get_contiguous_indexes(self.ptr, n)

        try:
            self.data[indexes] = batch
        except ValueError:
            if self.data.is_empty():
                self.data = _create_value(batch[0], self.capacity, stack=True)
            else:
                _alloc_by_keys_diff(self.data, batch[0], self.capacity, stack=True)
            self.data[indexes] = batch

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)


def _get_sample_sizes(target: int, sizes: List[int]) -> List[int]:
    total = sum(sizes)
    sample_sizes = [math.floor(size * target / total) for size in sizes]
    diff = target - sum(sample_sizes)
    for i in range(len(sizes)):
        if diff > 0 and sample_sizes[i] < sizes[i]:
            diff -= 1
            sample_sizes[i] += 1

    assert sum(sample_sizes) == target
    return sample_sizes


class ReplayMemoryManager(object):
    def __init__(self, memories: List[ReplayMemory]) -> None:
        self.memories = memories

    @property
    def num_memories(self) -> int:
        return len(self.memories)

    def store(self, batches: List[Batch], ids: List[int]) -> None:
        for i, batch in zip(ids, batches):
            self.memories[i].store(batch)

    def sample(
        self,
        size: Optional[int] = None,
        ids: Optional[List[int]] = None,
        random: bool = False,
        pop: bool = False,
    ) -> Dict[int, Batch]:
        if ids is None:
            ids = list(range(self.num_memories))

        if size is None:
            return {
                i: self.memories[i].sample(size=None, random=random, pop=pop) for i in ids if self.memories[i].size > 0
            }
        else:
            sizes = [self.memories[i].size for i in ids]
            assert 0 < size <= sum(sizes)
            sample_sizes = _get_sample_sizes(size, sizes)

            batch_dict = {
                i: self.memories[i].sample(size=sample_size, random=random, pop=pop)
                for i, sample_size in zip(ids, sample_sizes)
                if sample_size > 0
            }
            assert sum(len(v) for v in batch_dict.values()) == size
            return batch_dict
