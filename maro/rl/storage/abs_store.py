# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Sequence


class AbsStore(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get(self, indexes: Sequence):
        """
        Get contents.

        Args:
            indexes: a sequence of indexes where store contents are to be retrieved.
        Returns:
            retrieved contents.
        """
        pass

    @abstractmethod
    def put(self, contents: Sequence, index_sampler: Sequence):
        """
        Put new contents.

        Args:
            contents: Item object list.
            index_sampler: optional custom sampler used to obtain indexes for overwriting
        Returns:
            The newly appended item indexes.
        """
        pass

    @abstractmethod
    def update(self, indexes: Sequence, contents: Sequence):
        """
        Update selected contents.

        Args:
            indexes: Item indexes list.
            contents: Item list, which has the same length as indexes.
        Returns:
            The updated item index list.
        """
        pass

    def filter(self, filters):
        """
        Multi-filter method.
            The next layer filter input is the last layer filter output.

        Args:
            filters (Iterable[Callable]): Filter list, each item is a lambda function.
                The lambda input is a tuple, (index, object).
                i.e. [lambda x: x['a'] == 1 and x['b'] == 1]

        Returns:
            list: Filtered indexes or contents. i.e. [1, 2, 3], ['a', 'b', 'c']
        """
        pass

    @abstractmethod
    def sample(self, size, weights: Sequence, replace: bool = True):
        """
        Obtain a random sample from the experience pool.

        Args:
            size (int): sample sizes for each round of sampling in the chain. If this is a single integer, it is
                        used as the sample size for all samplers in the chain.
            weights (Sequence): a sequence of sampling weights
            replace (bool): if True, sampling is performed with replacement. Default is True.
        Returns:
            Tuple: Sampled indexes and contents.i.e. [1, 2, 3], ['a', 'b', 'c']
        """
        pass
