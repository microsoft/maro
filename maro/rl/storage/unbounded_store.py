# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from typing import Callable, Tuple, List

from .abstract_store import AbstractStore


class UnboundedStore(AbstractStore):
    def __init__(self):
        super().__init__()
        self._list = []
        self._size = 0

    def get(self, indexes: [int]) -> list:
        return [self._list[i] for i in indexes]

    def put(self, items: list, index_sampler: Callable[[object], float] = None) -> List[int]:
        self._list.extend(items)
        self._size += len(items)
        return list(range(len(self._list)-len(items), len(self._list)))

    def update(self, indexes: [int], items: list, key=None) -> List[int]:
        assert(len(indexes) == len(items))
        for idx, item in zip(indexes, items):
            if key is not None:
                self._list[idx][key] = item
            else:
                self._list[idx] = item
        return indexes

    def apply_multi_filters(self, filters: [Callable]) -> list:
        lst = self._list
        for f in filters:
            lst = list(filter(f, lst))

        return lst

    def apply_multi_samplers(self, samplers: [Tuple[Callable, int]]) -> Tuple[list, list]:
        indexes = range(self._size)
        for weight_fn, sample_size in samplers:
            weights = [weight_fn(self._list[i]) for i in indexes]
            indexes = random.choices(indexes, weights=weights, k=sample_size)

        return indexes, [self._list[i] for i in indexes]

    def take(self):
        return self._list[:]

    def clear(self):
        self._list = []
        self._size = 0

    @property
    def size(self):
        return self._size
