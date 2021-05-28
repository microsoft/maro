# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

from .experience_manager import ExperienceManager, ExperienceSet


class AbsSampler(ABC):
    def __init__(self, experience_manager: ExperienceManager):
        super().__init__()
        self.experience_manager = experience_manager

    @abstractmethod
    def get(self) -> ExperienceSet:
        raise NotImplementedError
