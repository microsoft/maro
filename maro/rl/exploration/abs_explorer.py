# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsExplorer(ABC):
    """Abstract explorer class for generating exploration rates.

    """
    def __init__(self):
        pass

    @abstractmethod
    def set_parameters(self, **exploration_params):
        return NotImplementedError

    @abstractmethod
    def __call__(self, action):
        return NotImplementedError
