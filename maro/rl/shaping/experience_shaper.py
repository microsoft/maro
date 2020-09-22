# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Union
from .abs_shaper import AbsShaper


class ExperienceShaper(AbsShaper):
    """
    A reward shaper is used to record transitions during a roll-out episode and perform necessary post-processing
    at the end of the episode. The post-processing logic is encapsulated in the abstract shape() method and needs
    to be implemented for each scenario. In particular, it is necessary to specify how to determine the reward for
    an action given the business metrics associated with the corresponding transition.
    """
    def __init__(self, reward_func: Union[Callable, None], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reward_func = reward_func

    @abstractmethod
    def __call__(self, trajectory, snapshot_list) -> Iterable:
        """
        Converts transitions along a trajectory to experiences.

        Args:
            snapshot_list: snapshot list stored in the env at the end of an episode.
        Returns:
            Experiences that can be used by the algorithm
        """
        pass

    def reset(self):
        pass
