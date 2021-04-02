# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Callable, Sequence


class AbsStore(ABC):
    """A data store abstraction that supports get, put, update and sample operations."""
    def __init__(self):
        pass

    @abstractmethod
    def get(self, indexes: Sequence):
        """Get contents.

        Args:
            indexes: A sequence of indexes to retrieve contents at.
        Returns:
            Retrieved contents.
        """
        pass

    def put(self, contents: Sequence):
        """Put new contents.

        Args:
            contents (Sequence): Contents to be added to the store.
        Returns:
            The indexes where the newly added entries reside in the store.
        """
        pass

    @abstractmethod
    def update(self, indexes: Sequence, contents: Sequence):
        """Update the store contents at given positions.

        Args:
            indexes (Sequence): Positions where updates are to be made.
            contents (Sequence): Item list, which has the same length as indexes.
        Returns:
            The indexes where store contents are updated.
        """
        pass

    def filter(self, filters: Sequence[Callable]):
        """Multi-filter method.

        The input to one filter is the output from the previous filter.

        Args:
            filters (Sequence[Callable]): Filter list, each item is a lambda function,
                                          e.g., [lambda d: d['a'] == 1 and d['b'] == 1].
        Returns:
            Filtered indexes and corresponding objects.
        """
        pass
