# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .experience_memory import ExperienceMemory


class AbsSampler(ABC):
    def __init__(self, data: ExperienceMemory, batch_size: int):
        self.data = data
        self.batch_size = batch_size

    @abstractmethod
    def sample(self) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def update(self):
        """Update statistics used for sampling."""
        pass


class UniformSampler(AbsSampler):
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
    def __init__(self, data: ExperienceMemory, batch_size: int, replace: bool = True):
        super().__init__(data, batch_size)
        self.replace = replace

    def sample(self):
        indexes = np.random.choice(len(self.data), size=self.batch_size, replace=self.replace)
        return indexes

    def update(self, indexes, values):
        pass
