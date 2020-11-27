# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Union

from maro.utils.exception.rl_toolkit_exception import InfiniteTrainingLoopError, InvalidEpisodeError

from .exploration_parameter_generator import (
    DynamicExplorationParameterGenerator, StaticExplorationParameterGenerator
)


class Scheduler(object):
    """Scheduler that generates exploration parameters for each episode.

    Args:
        max_ep (int): Maximum number of episodes to be run.
    """

    def __init__(
        self,
        max_ep: int,
        warmup_ep: int = 0,
        early_stopping_callback: Callable = None,
        exploration_parameter_generator_cls=None,
        exploration_parameter_generator_config: dict = None
    ):
        if max_ep < -1:
            raise InvalidEpisodeError("max_episode can only be a non-negative integer or -1.")
        if max_ep == -1 and early_stopping_callback is None:
            raise InfiniteTrainingLoopError(
                "The training loop will run forever since neither maximum episode nor early stopping checker "
                "is provided. "
            )
        self._max_ep = max_ep
        self._warmup_ep = warmup_ep
        self._early_stopping_callback = early_stopping_callback
        self._current_ep = 0
        self._performance_history = []
        if exploration_parameter_generator_cls is None:
            self._exploration_parameter_generator = None
        elif issubclass(exploration_parameter_generator_cls, StaticExplorationParameterGenerator):
            self._exploration_parameter_generator = exploration_parameter_generator_cls(
                max_ep, **exploration_parameter_generator_config
            )
        else:
            self._exploration_parameter_generator = exploration_parameter_generator_cls(
                **exploration_parameter_generator_config
            )

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_ep == self._max_ep:
            raise StopIteration
        if self._current_ep >= self._warmup_ep:
            if self._early_stopping_callback and self._early_stopping_callback(self._performance_history):
                raise StopIteration
        self._current_ep += 1
        if isinstance(self._exploration_parameter_generator, StaticExplorationParameterGenerator):
            return self._exploration_parameter_generator.next()
        elif isinstance(self._exploration_parameter_generator, DynamicExplorationParameterGenerator):
            return self._exploration_parameter_generator.next(self._performance_history)

    @property
    def current_ep(self):
        return self._current_ep

    def record_performance(self, performance):
        self._performance_history.append(performance)
