# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Callable, List, Sequence, Tuple

import numpy as np

from .abs_store import AbsStore
from .utils import check_uniformity, get_update_indexes, normalize, OverwriteType
from maro.utils import clone


class ColumnBasedStore(AbsStore):
    def __init__(self, capacity: int = -1, overwrite_type: OverwriteType = None):
        """
        A ColumnBasedStore instance that uses a Python list as its internal storage data structure and supports unlimited
        and limited storage.

        Args:
            capacity: if -1, the store is of unlimited capacity. Default is -1.
            overwrite_type (OverwriteType): If storage capacity is bounded, this specifies how existing entries
                                            are overwritten.
        """
        super().__init__()
        self._capacity = capacity
        self._store = defaultdict(lambda: [] if self._capacity < 0 else [None] * self._capacity)
        self._size = 0
        self._overwrite_type = overwrite_type
        self._iter_index = 0

    def __len__(self):
        return self._size

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_index >= self._size:
            self._iter_index = 0
            raise StopIteration
        index = self._iter_index
        self._iter_index += 1
        return {k: lst[index] for k, lst in self._store.items()}

    def __getitem__(self, index: int):
        return {k: lst[index] for k, lst in self._store.items()}

    @property
    def capacity(self):
        return self._capacity

    @property
    def overwrite_type(self):
        return self._overwrite_type

    def get(self, indexes: [int]) -> dict:
        return {k: [self._store[k][i] for i in indexes] for k in self._store}

    @check_uniformity(arg_num=1)
    def put(self, contents: dict, overwrite_indexes: Sequence = None) -> List[int]:
        if len(self._store) > 0 and contents.keys() != self._store.keys():
            raise ValueError(f"expected keys {list(self._store.keys())}, got {list(contents.keys())}")
        added_size = len(contents[next(iter(contents))])
        if self._capacity < 0:
            for key, lst in contents.items():
                self._store[key].extend(lst)
            self._size += added_size
            return list(range(self._size-added_size, self._size))
        else:
            write_indexes = get_update_indexes(self._size, added_size, self._capacity, self._overwrite_type,
                                               overwrite_indexes=overwrite_indexes)
            self.update(write_indexes, contents)
            self._size = min(self._capacity, self._size + added_size)
            return write_indexes

    @check_uniformity(arg_num=2)
    def update(self, indexes: Sequence, contents: dict) -> Sequence:
        """
        Update selected contents.

        Args:
            indexes: Item indexes list.
            contents: contents to write to the internal store at given positions
        Returns:
            The updated item indexes.
        """
        for key, value_list in contents.items():
            assert len(indexes) == len(value_list), f"expected updates at {len(indexes)} indexes, got {len(value_list)}"
            for index, value in zip(indexes, value_list):
                self._store[key][index] = value

        return indexes

    def apply_multi_filters(self, filters: Sequence[Callable]):
        """Multi-filter method.
            The next layer filter input is the last layer filter output.

        Args:
            filters (Sequence[Callable]): Filter list, each item is a lambda function.
                                          i.e. [lambda d: d['a'] == 1 and d['b'] == 1]
        Returns:
            Filtered indexes and corresponding objects.
        """
        indexes = range(self._size)
        for f in filters:
            indexes = [i for i in indexes if f(self[i])]

        return indexes, self.get(indexes)

    def apply_multi_samplers(self, samplers: Sequence[Tuple[Callable, int]], replace: bool = True) -> Tuple:
        """Multi-samplers method.
            The next layer sampler input is the last layer sampler output.

        Args:
            samplers ([Tuple[Callable, int]]): Sampler list, each sampler is a tuple.
                The 1st item of the tuple is a lambda function.
                    The 1st lambda input is index, the 2nd lambda input is a object.
                The 2nd item of the tuple is the sample size.
                i.e. [(lambda o: o['a'], 3)]
            replace: If True, sampling will be performed with replacement.
        Returns:
            Sampled indexes and corresponding objects.
        """
        indexes = range(self._size)
        for weight_fn, sample_size in samplers:
            weights = np.asarray([weight_fn(self[i]) for i in indexes])
            indexes = np.random.choice(indexes, size=sample_size, replace=replace, p=weights/np.sum(weights))

        return indexes, self.get(indexes)

    @normalize
    def sample(self, size, weights: Sequence = None, replace: bool = True):
        """
        Obtain a random sample from the experience pool.

        Args:
            size (int): sample sizes for each round of sampling in the chain. If this is a single integer, it is
                        used as the sample size for all samplers in the chain.
            weights (Sequence): a sequence of sampling weights.
            replace (bool): if True, sampling is performed with replacement. Default is True.
        Returns:
            Sampled indexes and the corresponding objects.
        """
        indexes = np.random.choice(self._size, size=size, replace=replace, p=weights)
        return indexes, self.get(indexes)

    def sample_by_key(self, key, size, replace: bool = True):
        """
        Obtain a random sample from the store using one of the columns as sampling weights.

        Args:
            key: the column whose values are to be used as sampling weights.
            size: sample size.
            replace: If True, sampling is performed with replacement.
        Returns:
            Sampled indexes and the corresponding objects.
        """
        weights = np.asarray(self._store[key][:self._size] if self._size < self._capacity else self._store[key])
        indexes = np.random.choice(self._size, size=size, replace=replace, p=weights/np.sum(weights))
        return indexes, self.get(indexes)

    def sample_by_keys(self, keys: Sequence, sizes: Sequence, replace: bool = True):
        """
        Obtain a random sample from the store by chained sampling using multiple columns as sampling weights.

        Args:
            keys: the column whose values are to be used as sampling weights.
            sizes: sample size.
            replace: If True, sampling is performed with replacement.
        Returns:
            Sampled indexes and the corresponding objects.
        """
        if len(keys) != len(sizes):
            raise ValueError(f"expected sizes of length {len(keys)}, got {len(sizes)}")

        indexes = range(self._size)
        for key, size in zip(keys, sizes):
            weights = np.asarray([self._store[key][i] for i in indexes])
            indexes = np.random.choice(indexes, size=size, replace=replace, p=weights/np.sum(weights))

        return indexes, self.get(indexes)

    def dumps(self):
        return clone(dict(self._store))

    def get_by_key(self, key):
        return self._store[key]

    def clear(self):
        del self._store
        self._store = defaultdict(lambda: [] if self._capacity < 0 else [None] * self._capacity)
        self._size = 0
        self._iter_index = 0
