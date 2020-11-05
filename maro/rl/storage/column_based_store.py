# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Callable, List, Sequence, Tuple

import numpy as np

from maro.utils import clone

from .abs_store import AbsStore
from .utils import OverwriteType, check_uniformity, get_update_indexes, normalize


class ColumnBasedStore(AbsStore):
    """
    An implementation of ``AbsStore`` for experience storage in RL.

    This implementation uses a dictionary of lists as the internal data structure. The objects for each key
    are stored in a list. To be useful for experience storage in RL, uniformity checks are performed during
    put operations to ensure that the list lengths stay the same for all keys at all times. Both unlimited
    and limited storage are supported.

    Args:
        capacity (int): If negative, the store is of unlimited capacity. Defaults to -1.
        overwrite_type (OverwriteType): If storage capacity is bounded, this specifies how existing entries
            are overwritten when the capacity is exceeded. Two types of overwrite behavior are supported:
            - Rolling, where overwrite occurs sequentially with wrap-around.
            - Random, where overwrite occurs randomly among filled positions.
            Alternatively, the user may also specify overwrite positions (see ``put``).
    """
    def __init__(self, capacity: int = -1, overwrite_type: OverwriteType = None):
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

    def __getstate__(self):
        """A patch for picking the object with lambda.
        Using the default ``__dict__`` would make the object unpicklable due to the lambda function involved in the
        ``defaultdict`` definition of the ``_store`` attribute.
        """
        obj_dict = self.__dict__
        obj_dict["_store"] = dict(obj_dict["_store"])
        return obj_dict

    @property
    def capacity(self):
        """Store capacity.

        If negative, the store grows without bound. Otherwise, the number of items in the store will not exceed
        this capacity.
        """
        return self._capacity

    @property
    def overwrite_type(self):
        """An ``OverwriteType`` member indicating the overwrite behavior when the store capacity is exceeded."""
        return self._overwrite_type

    def get(self, indexes: [int]) -> dict:
        return {k: [self._store[k][i] for i in indexes] for k in self._store}

    @check_uniformity(arg_num=1)
    def put(self, contents: dict, overwrite_indexes: Sequence = None) -> List[int]:
        """Put new contents in the store.

        Args:
            contents (Sequence): Item object list.
            overwrite_indexes (Sequence, optional): indexes where the contents are to be overwritten. This is only
                used when the store has a fixed capacity and putting ``contents`` in the store would exceed this
                capacity. If this is None and overwriting is necessary, rolling or random overwriting will be done
                according to the ``overwrite`` property. Defaults to None.
        Returns:
            The indexes where the newly added entries reside in the store.
        """
        if len(self._store) > 0 and contents.keys() != self._store.keys():
            raise ValueError(f"expected keys {list(self._store.keys())}, got {list(contents.keys())}")
        added_size = len(contents[next(iter(contents))])
        if self._capacity < 0:
            for key, lst in contents.items():
                self._store[key].extend(lst)
            self._size += added_size
            return list(range(self._size - added_size, self._size))
        else:
            write_indexes = get_update_indexes(
                self._size, added_size, self._capacity, self._overwrite_type, overwrite_indexes=overwrite_indexes
            )
            self.update(write_indexes, contents)
            self._size = min(self._capacity, self._size + added_size)
            return write_indexes

    @check_uniformity(arg_num=2)
    def update(self, indexes: Sequence, contents: dict) -> Sequence:
        """
        Update contents at given positions.

        Args:
            indexes (Sequence): Positions where updates are to be made.
            contents (dict): Contents to write to the internal store at given positions. It is subject to uniformity
                checks to ensure that the lists for all keys have the same length.

        Returns:
            The indexes where store contents are updated.
        """
        for key, value_list in contents.items():
            assert len(indexes) == len(value_list), f"expected updates at {len(indexes)} indexes, got {len(value_list)}"
            for index, value in zip(indexes, value_list):
                self._store[key][index] = value

        return indexes

    def apply_multi_filters(self, filters: Sequence[Callable]):
        """Multi-filter method.

            The input to one filter is the output from its predecessor in the sequence.

        Args:
            filters (Sequence[Callable]): Filter list, each item is a lambda function,
                e.g., [lambda d: d['a'] == 1 and d['b'] == 1].

        Returns:
            Filtered indexes and corresponding objects.
        """
        indexes = range(self._size)
        for f in filters:
            indexes = [i for i in indexes if f(self[i])]

        return indexes, self.get(indexes)

    def apply_multi_samplers(self, samplers: Sequence, replace: bool = True) -> Tuple:
        """Multi-samplers method.

        This implements chained sampling where the input to one sampler is the output from its predecessor in
        the sequence.

        Args:
            samplers (Sequence): A sequence of weight functions for computing the sampling weights of the items
                in the store,
                e.g., [lambda d: d['a'], lambda d: d['b']].
            replace (bool): If True, sampling will be performed with replacement.
        Returns:
            Sampled indexes and corresponding objects.
        """
        indexes = range(self._size)
        for weight_fn, sample_size in samplers:
            weights = np.asarray([weight_fn(self[i]) for i in indexes])
            indexes = np.random.choice(indexes, size=sample_size, replace=replace, p=weights / np.sum(weights))

        return indexes, self.get(indexes)

    @normalize
    def sample(self, size, weights: Sequence = None, replace: bool = True):
        """
        Obtain a random sample from the experience pool.

        Args:
            size (int): sample sizes for each round of sampling in the chain. If this is a single integer, it is
                        used as the sample size for all samplers in the chain.
            weights (Sequence): a sequence of sampling weights.
            replace (bool): if True, sampling is performed with replacement. Defaults to True.
        Returns:
            Sampled indexes and the corresponding objects,
            e.g., [1, 2, 3], ['a', 'b', 'c'].
        """
        indexes = np.random.choice(self._size, size=size, replace=replace, p=weights)
        return indexes, self.get(indexes)

    def sample_by_key(self, key, size: int, replace: bool = True):
        """
        Obtain a random sample from the store using one of the columns as sampling weights.

        Args:
            key: the column whose values are to be used as sampling weights.
            size (int): sample size.
            replace (bool): If True, sampling is performed with replacement.
        Returns:
            Sampled indexes and the corresponding objects.
        """
        weights = np.asarray(self._store[key][:self._size] if self._size < self._capacity else self._store[key])
        indexes = np.random.choice(self._size, size=size, replace=replace, p=weights / np.sum(weights))
        return indexes, self.get(indexes)

    def sample_by_keys(self, keys: Sequence, sizes: Sequence, replace: bool = True):
        """
        Obtain a random sample from the store by chained sampling using multiple columns as sampling weights.

        Args:
            keys (Sequence): the column whose values are to be used as sampling weights.
            sizes (Sequence): sample size.
            replace (bool): If True, sampling is performed with replacement.
        Returns:
            Sampled indexes and the corresponding objects.
        """
        if len(keys) != len(sizes):
            raise ValueError(f"expected sizes of length {len(keys)}, got {len(sizes)}")

        indexes = range(self._size)
        for key, size in zip(keys, sizes):
            weights = np.asarray([self._store[key][i] for i in indexes])
            indexes = np.random.choice(indexes, size=size, replace=replace, p=weights / np.sum(weights))

        return indexes, self.get(indexes)

    def dumps(self):
        """Return a deep copy of store contents."""
        return clone(dict(self._store))

    def get_by_key(self, key):
        """Get the contents of the store corresponding to ``key``."""
        return self._store[key]

    def clear(self):
        """Empty the store."""
        del self._store
        self._store = defaultdict(lambda: [] if self._capacity < 0 else [None] * self._capacity)
        self._size = 0
        self._iter_index = 0
