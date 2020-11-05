# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from .abs_shaper import AbsShaper


class ActionShaper(AbsShaper):
    """Action shaper class.

    An action shaper is used to convert an agent's model output to an environment executable action.
    """
    @abstractmethod
    def __call__(self, model_action, decision_event, snapshot_list):
        """This method converts a model output to an environment executable action.

        Information from the decision event and snapshot list may also be needed as contexts for this conversion.
        """
        pass

    def reset(self):
        """Reset stateful members, if any, to their states at the beginning of an episode."""
        pass
