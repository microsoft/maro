# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsEarlyStoppingChecker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, performance_history):
        return NotImplemented
