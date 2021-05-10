# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Callable


class AbsExploration(ABC):
    """Abstract exploration class for generating exploration rates.

    """
    def __init__(self):
        self.scheduler = {}

    def register_schedule(
        self,
        scheduler_cls,
        param_name: str,
        last_ep: int,
        initial_value=None,
        **kwargs
    ):
        self.scheduler[param_name] = scheduler_cls(self, param_name, last_ep, initial_value=initial_value, **kwargs)

    @abstractmethod
    def __call__(self, action):
        return NotImplementedError 

    @property
    def parameters(self):
        return {param_name: getattr(self, param_name) for param_name in self.scheduler}

    def step(self):
        for scheduler in self.scheduler.values():
            scheduler.step()


class NullExploration(AbsExploration):
    def __init__(self):
        pass

    def __call__(self, action):
        return action
