# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Callable, Tuple


class AbstractStore(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def put(self, items, index_sampler: Callable[[object], float] = None):
        """
        Put new items.

        Args:
            items: Item object list.
            index_sampler: optional custom sampler used to obtain indexes for overwriting
        Returns:
            The new appended item indexes.
        """
        return NotImplementedError

    @abstractmethod
    def get(self, indexes):
        """
        Get items.

        Args:
            indexes: Item indexes list.
        Returns:
            The got item list.
        """
        return NotImplementedError

    @abstractmethod
    def update(self, indexes, items, key=None):
        """
        Update selected items.

        Args:
            indexes: Item indexes list.
            items: Item list, which has the same length as indexes.
            key: If specified, the corresponding values will be replaced by items
        Returns:
            The updated item index list.
        """
        return NotImplementedError

    @abstractmethod
    def apply_multi_filters(self, filters: [Callable]):
        """
        Multi-filter method.
            The next layer filter input is the last layer filter output.

        Args:
            filters ([Callable[[Tuple[int, object]], Tuple[int, object]]]): Filter list, each item is a lambda function. \n
                The lambda input is a tuple, (index, object). \n
                i.e. [lambda x: x['a'] == 1 and x['b'] == 1]

        Returns:
            list: Filtered indexes or items. i.e. [1, 2, 3], ['a', 'b', 'c']
        """
        return NotImplementedError

    @abstractmethod
    def apply_multi_samplers(self, samplers: [Tuple[Callable, int]]):
        """
        Multi-samplers method.
            The next layer sampler input is the last layer sampler output.

        Args:
            samplers ([Tuple[Callable[[int, object], Tuple[int, object]], int]]): Sampler list, each sampler is a tuple. \n
                The 1st item of the tuple is a lambda function. \n
                    The 1st lambda input is index, the 2nd lambda input is a object. \n
                The 2nd item of the tuple is the sample size. \n
                i.e. [(lambda i, o: (i, o['a']), 3)]

        Returns:
            list: Sampled indexes or items.i.e. [1, 2, 3], ['a', 'b', 'c']
        """
        return NotImplementedError

    @abstractmethod
    def take(self):
        """
        Return a copy of the store contents
        """
        return NotImplementedError

    @abstractmethod
    def clear(self):
        """
        Clears the store
        """
        return NotImplementedError
