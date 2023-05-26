# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import warnings
from typing import Dict, List, Optional, Tuple

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

    def get_all_indexes(self) -> np.ndarray:
        return self._get_contiguous_indexes(self.header, self.size)

    def sample_by_indexes(self, indexes: np.ndarray) -> Batch:
        return self.data[indexes]

    def sample(self, size: Optional[int] = None, random: bool = False, pop: bool = False) -> Batch:
        if size is None:
            indexes = self.get_all_indexes()
            batch = self.data[indexes]
            if pop:
                self.reset()
            return batch
        else:
            if random:
                assert not pop, "Pop is not allowed under random mode."
                indexes = np.random.choice(np.arange(self.size), size=size, replace=True)
                return self.data[indexes]
            else:
                assert 0 < size <= self.size
                indexes = self._get_contiguous_indexes(self.header, size)
                batch = self.data[indexes]
                if pop:
                    self.size -= size  # Reduce self_size only. Pop elements will not affect self._ptr.
                return batch

    def store(self, batch: Batch) -> np.ndarray:
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

        return indexes


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
    def __init__(
        self,
        memories: List[ReplayMemory],
        priority_params: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.memories = memories

        if priority_params is not None:
            self.maintain_priority = True
            self.alpha, self.beta = priority_params
            self.max_prio = self.min_prio = 1.0
            self.weights = np.zeros(sum(m.capacity for m in memories), dtype=np.float32)
        else:
            self.maintain_priority = False

    def init_weights(self, indexes: np.ndarray) -> None:
        self.weights[indexes] = self.max_prio * self.alpha

    def get_weights(self, indexes: np.ndarray) -> np.ndarray:
        return (self.weights[indexes] / self.min_prio) ** -self.beta

    def update_weights(self, indexes: np.ndarray, weights: np.ndarray) -> None:
        assert indexes.shape == weights.shape

        weights = np.abs(weights) + np.finfo(np.float32).eps.item()
        self.weights[indexes] = weights**self.alpha
        self.max_prio = max(self.max_prio, weights.max())
        self.min_prio = min(self.min_prio, weights.min())

    @property
    def n_sample(self) -> int:
        return sum(memory.size for memory in self.memories)

    @property
    def num_memories(self) -> int:
        return len(self.memories)

    def store(self, batches: List[Batch], ids: List[int]) -> None:
        store_indexes = [self.memories[i].store(batch) for i, batch in zip(ids, batches)]

        if self.maintain_priority:
            self.init_weights(indexes=self._encode_indexes(store_indexes))

    def _get_offsets(self) -> List[int]:
        offsets = [0]
        for m in self.memories:
            offsets.append(offsets[-1] + m.capacity)
        return offsets

    def _encode_indexes(self, groups: List[np.ndarray]) -> np.ndarray:
        offsets = self._get_offsets()
        indexes = np.concatenate([o + i for o, i in zip(offsets[:-1], groups)])
        return indexes

    def _decode_indexes(self, indexes: np.ndarray) -> List[np.ndarray]:
        offsets = self._get_offsets()
        indexes = np.sort(indexes)
        groups = []
        i = 0
        for lower, upper in zip(offsets[:-1], offsets[1:]):
            j = i
            while j < len(indexes) and indexes[j] < upper:
                j += 1
            groups.append(indexes[i:j] - lower)
            i = j
        return groups

    def sample_random_indexes(self, size: int, weighted: bool = False) -> np.ndarray:
        indexes = self._encode_indexes([m.get_all_indexes() for m in self.memories])

        if weighted:
            assert self.maintain_priority
            weights = self.weights[indexes]
            weights /= weights.sum()
            return np.random.choice(indexes, size=size, replace=True, p=weights)
        else:
            return np.random.choice(indexes, size=size, replace=True)

    def sample_by_indexes(self, indexes: np.ndarray) -> Batch:
        group_indexes = self._decode_indexes(indexes)
        batch_list = [m.sample_by_indexes(indexes) for m, indexes in zip(self.memories, group_indexes)]
        return Batch.cat(batch_list)

    def sample_separated(
        self,
        size: Optional[int] = None,
        random: bool = False,
        pop: bool = False,
    ) -> Dict[int, Batch]:
        ids = list(range(self.num_memories))

        if size is None:
            return {
                i: self.memories[i].sample(size=None, random=random, pop=pop) for i in ids if self.memories[i].size > 0
            }
        else:
            sizes = [self.memories[i].size for i in ids]
            if not random:
                assert 0 < size <= sum(sizes)
            sample_sizes = _get_sample_sizes(size, sizes)

            batch_dict = {
                i: self.memories[i].sample(size=sample_size, random=random, pop=pop)
                for i, sample_size in zip(ids, sample_sizes)
                if sample_size > 0
            }
            assert sum(len(v) for v in batch_dict.values()) == size
            return batch_dict

    def sample(
        self,
        size: Optional[int] = None,
        random: bool = False,
        pop: bool = False,
    ) -> Batch:
        batch_dict = self.sample_separated(size, random, pop)
        batch_list = list(batch_dict.values())
        return Batch.cat(batch_list)
