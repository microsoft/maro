# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

import numpy as np

from .experience_memory import ExperienceMemory, ExperienceSet


class AbsSampler(ABC):
    def __init__(self, experience_memory: ExperienceMemory):
        super().__init__()
        self.experience_memory = experience_memory

    @abstractmethod
    def get(self) -> ExperienceSet:
        raise NotImplementedError


class UniformSampler(AbsSampler):
    def __init__(self, experience_memory: ExperienceMemory, batch_size: int, replace: bool = True):
        super().__init__(experience_memory)
        self.batch_size = batch_size
        self.replace = replace

    def get(self) -> ExperienceSet:
        indexes = np.random.choice(self.experience_memory.size, size=self.batch_size, replace=self.replace)
        return ExperienceSet(
            *[[self.experience_memory.data[key][idx] for idx in indexes] for key in self.experience_memory.keys]
        )
