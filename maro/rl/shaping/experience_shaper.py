# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Callable, Iterable, Sequence, Union

from .abs_shaper import AbsShaper


class ExperienceShaper(AbsShaper):
    """Experience shaper class.

    An experience shaper is used to convert a trajectory of transitions to experiences for training.
    """
    def __init__(self, reward_func: Union[Callable, None], *args, **kwargs):
        """
        Args:
            reward_func (Callable or None): A reward function to compute immediate rewards from the business
                metrics associated with a transition. Under certain circumstances, reward calculations may
                need access to stateful objects within the shaper, in which case users are free to implement
                their own methods for calculating rewards.
        """
        super().__init__(*args, **kwargs)
        self._reward_func = reward_func

    @abstractmethod
    def __call__(self, trajectory: Sequence, snapshot_list) -> Iterable:
        """Converts transitions along a trajectory to experiences.

        Args:
            trajectory(Sequence): A sequence of transitions recorded by the agent manager during roll-out.
            snapshot_list: Snapshot list stored in the environment at the end of an episode.
        Returns:
            Experiences that can be used by the algorithm.
        """
        pass

    def reset(self):
        """Reset stateful members, if any, to their states at the beginning of an episode."""
        pass
