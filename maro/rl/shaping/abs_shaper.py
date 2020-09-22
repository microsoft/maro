# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsShaper(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
