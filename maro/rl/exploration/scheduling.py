# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import List, Tuple


class AbsExplorationScheduler(ABC):
    """Abstract exploration scheduler.

    Args:
        exploration_params (dict): The exploration params attribute from some ``RLPolicy`` instance to which the
            scheduler is applied.
        param_name (str): Name of the exploration parameter to which the scheduler is applied.
        initial_value (float, default=None): Initial value for the exploration parameter. If None, the value used
            when instantiating the policy will be used as the initial value.
    """

    def __init__(self, exploration_params: dict, param_name: str, initial_value: float = None) -> None:
        super().__init__()
        self._exploration_params = exploration_params
        self.param_name = param_name
        if initial_value is not None:
            self._exploration_params[self.param_name] = initial_value

    def get_value(self) -> float:
        return self._exploration_params[self.param_name]

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError


class LinearExplorationScheduler(AbsExplorationScheduler):
    """Linear exploration parameter schedule.

    Args:
        exploration_params (dict): The exploration params attribute from some ``RLPolicy`` instance to which the
            scheduler is applied.
        param_name (str): Name of the exploration parameter to which the scheduler is applied.
        last_ep (int): Last episode.
        final_value (float): The value of the exploration parameter corresponding to ``last_ep``.
        start_ep (int, default=1): starting episode.
        initial_value (float, default=None): Initial value for the exploration parameter. If None, the value used
            when instantiating the policy will be used as the initial value.
    """

    def __init__(
        self,
        exploration_params: dict,
        param_name: str,
        *,
        last_ep: int,
        final_value: float,
        start_ep: int = 1,
        initial_value: float = None,
    ) -> None:
        super().__init__(exploration_params, param_name, initial_value=initial_value)
        self.final_value = final_value
        if last_ep > 1:
            self.delta = (self.final_value - self._exploration_params[self.param_name]) / (last_ep - start_ep)
        else:
            self.delta = 0

    def step(self) -> None:
        if self._exploration_params[self.param_name] == self.final_value:
            return

        self._exploration_params[self.param_name] += self.delta


class MultiLinearExplorationScheduler(AbsExplorationScheduler):
    """Exploration parameter schedule that consists of multiple linear phases.

    Args:
        exploration_params (dict): The exploration params attribute from some ``RLPolicy`` instance to which the
            scheduler is applied.
        param_name (str): Name of the exploration parameter to which the scheduler is applied.
        splits (List[Tuple[int, float]]): List of points that separate adjacent linear phases. Each
            point is a (episode, parameter_value) tuple that indicates the end of one linear phase and
            the start of another. These points do not have to be given in any particular order. There
            cannot be two points with the same first element (episode), or a ``ValueError`` will be raised.
        last_ep (int): Last episode.
        final_value (float): The value of the exploration parameter corresponding to ``last_ep``.
        start_ep (int, default=1): starting episode.
        initial_value (float, default=None): Initial value for the exploration parameter. If None, the value from
            the original dictionary the policy is instantiated with will be used as the initial value.
    """

    def __init__(
        self,
        exploration_params: dict,
        param_name: str,
        *,
        splits: List[Tuple[int, float]],
        last_ep: int,
        final_value: float,
        start_ep: int = 1,
        initial_value: float = None,
    ) -> None:
        super().__init__(exploration_params, param_name, initial_value=initial_value)

        # validate splits
        splits = [(start_ep, self._exploration_params[self.param_name])] + splits + [(last_ep, final_value)]
        splits.sort()
        for (ep, _), (ep2, _) in zip(splits, splits[1:]):
            if ep == ep2:
                raise ValueError("The zeroth element of split points must be unique")

        self.final_value = final_value
        self._splits = splits
        self._ep = start_ep
        self._split_index = 1
        self._delta = (self._splits[1][1] - self._exploration_params[self.param_name]) / (self._splits[1][0] - start_ep)

    def step(self) -> None:
        if self._split_index == len(self._splits):
            return

        self._exploration_params[self.param_name] += self._delta
        self._ep += 1
        if self._ep == self._splits[self._split_index][0]:
            self._split_index += 1
            if self._split_index < len(self._splits):
                self._delta = (self._splits[self._split_index][1] - self._splits[self._split_index - 1][1]) / (
                    self._splits[self._split_index][0] - self._splits[self._split_index - 1][0]
                )
