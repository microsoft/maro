# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from .abs_shaper import AbsShaper


class StateShaper(AbsShaper):
    """State shaper class.

    A state shaper is used to convert a decision event and snapshot list to model input.
    """
    @abstractmethod
    def __call__(self, decision_event, snapshot_list):
        pass

    def reset(self):
        """Reset stateful members, if any, to their states at the beginning of an episode."""
        pass
