# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

from maro.rl.actor import AbsActor
from maro.rl.scheduling import Scheduler


class AbsLearner(ABC):
    """Abstract learner class to control the policy learning process.

    Args:
        actor (AbsActor): An ``AbsActor`` instance that performs roll-outs.
        scheduler (Scheduler): A ``Scheduler`` instance that controls the training loop and parameter generation.
            Defaults to None.
    """
    def __init__(self, actor: AbsActor, scheduler: Scheduler = None):
        self.actor = actor
        self.scheduler = scheduler

    @abstractmethod
    def learn(self, *args, **kwargs):
        """The outermost training loop logic is implemented here."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """Train the underlying agents."""
        pass
