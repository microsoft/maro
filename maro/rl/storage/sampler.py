# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

import numpy as np

from .abs_store import AbsStore


class AbsSampler(ABC):
    def __init__(self, data: AbsStore):
        self.data = data

    @abstractmethod
    def sample(self, size: int):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        """Update statistics used for sampling."""
        pass


class UniformSampler(AbsSampler):
    def __init__(self, data, replace: bool = True):
        super().__init__(data)
        self.replace = replace

    def sample(self, size: int):
        """
        Obtain a random sample from the experience pool.

        Args:
            size (int): Sample sizes for each round of sampling in the chain. If this is a single integer, it is
                        used as the sample size for all samplers in the chain.
            weights (Union[list, np.ndarray]): Sampling weights.
            replace (bool): If True, sampling is performed with replacement. Defaults to True.
        Returns:
            Sampled indexes and the corresponding objects,
            e.g., [1, 2, 3], ['a', 'b', 'c'].
        """
        indexes = np.random.choice(len(self.data), size=size, replace=self.replace)
        return indexes, self.data.get(indexes=indexes)

    def update(self):
        pass


class PrioritizedSampler(AbsSampler):
    def __init__(self, data):
        super().__init__(data)
        
    def sample(self, size: int):
        pass

    def update(self):
        pass
