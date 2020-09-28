# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsShaper(ABC):
    def __init__(self, *args, **kwargs):
        """Abstract shaper class. Classes inheriting from it must implement the __call__ method.
        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """The general interface for conversion.
        """
        pass
