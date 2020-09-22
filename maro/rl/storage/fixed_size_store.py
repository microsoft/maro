# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import List, Callable, Tuple, Union
import numpy as np

from .abstract_store import AbstractStore


class OverwriteType(Enum):
    ROLLING = "rolling"
    RANDOM = "random"


class FixedSizeStore(AbstractStore):
    def __init__(self, capacity, overwrite_type):
        super().__init__()
        self._data = np.zeros(capacity, dtype=object)
        self._capacity = capacity
        self._overwrite_type = overwrite_type
        self._size = 0

    def put(self, items: Union[List, np.ndarray], index_sampler: Callable[[object], float] = None) -> np.ndarray:
        if len(items) > self._capacity:
            raise ValueError(f"Batch size should not exceed the store capacity.")

        num_overwrites = self._size + len(items) - self._capacity
        if num_overwrites <= 0:
            write_indexes = np.arange(self._size, self._size + len(items))
            self._size += len(items)
        else:  # num_overwrites old entries need to be overwritten
            if index_sampler is None:  # if no index sampler is provided, use the default overwrite rule
                start_index = self._size - self._capacity  # negative index convention
                if self._overwrite_type == OverwriteType.ROLLING:
                    write_indexes = np.arange(start_index, start_index + len(items))
                else:
                    write_indexes = np.concatenate([np.arange(self._size, self._capacity),
                                                    np.random.choice(self._size, replace=False, size=num_overwrites)]
                                                   )

            else:  # use custom index sampler
                weights = np.vectorize(index_sampler)(self._data[:self._size])
                weight_sum = np.sum(weights)
                write_indexes = np.concatenate([np.arange(self._size, self._capacity),
                                                np.random.choice(self._size,
                                                                 size=num_overwrites,
                                                                 replace=False,
                                                                 p=weights/weight_sum)]
                                               )

            self._size = self._capacity

        self._data[write_indexes] = items
        return np.asarray(write_indexes)

    def get(self, indexes: Union[List, np.ndarray]) -> np.ndarray:
        return self._data[indexes]

    def update(self, indexes: Union[List, np.ndarray], items: Union[List, np.ndarray], key=None) -> np.ndarray:
        assert(len(indexes) == len(items))
        if key is None:
            self._data[indexes] = items
        else:
            for idx, item in zip(indexes, items):
                self._data[idx][key] = item
        return np.asarray(indexes)

    def apply_multi_filters(self, filters: [Callable]) -> np.ndarray:
        result = self._data
        for ft in filters:
            result = result[np.vectorize(ft)(result)]

        return result

    def apply_multi_samplers(self, samplers: [Tuple[Callable, int]]) -> Tuple[np.ndarray, np.ndarray]:
        indexes = np.arange(self._size)
        for weight_fn, sample_size in samplers:
            weights = np.vectorize(weight_fn)(self._data[indexes])
            weight_sum = np.sum(weights)
            indexes = np.random.choice(indexes, size=sample_size, p=weights/weight_sum)

        return indexes, self._data[indexes]
    
    def take(self):
        return self._data.copy()

    def clear(self):
        """Clear current store.
        """
        self._data = np.zeros(self._capacity, dtype=object)
        self._size = 0

    @property
    def size(self):
        return self._size

    @property
    def capacity(self):
        return self._capacity
