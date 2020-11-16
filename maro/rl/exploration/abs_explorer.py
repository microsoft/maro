# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsExplorer(ABC):
    """Abstract explorer class for generating exploration rates.

    """
    def __init__(self):
        pass

    @abstractmethod
    def register_schedule(self, exploration_param_iter):
        return NotImplementedError

    @abstractmethod
    def load_exploration_params(self, exploration_params):
        return NotImplementedError

    @abstractmethod
    def dump_exploration_params(self):
        return NotImplementedError

    @abstractmethod
    def update(self):
        return NotImplementedError

    @abstractmethod
    def __call__(self, action):
        return NotImplementedError
