# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC


class AbsLearner(ABC):
    """Abstract learner class to control the policy learning process."""
    def __init__(self):
        pass

    def train(self, *args, **kwargs):
        """The outermost training loop logic is implemented here."""
        pass

    def test(self):
        """Test policy performance."""
        pass
