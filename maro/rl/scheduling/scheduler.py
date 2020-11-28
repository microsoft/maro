# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from maro.utils import DummyLogger, Logger
from maro.utils.exception.rl_toolkit_exception import InfiniteTrainingLoopError, InvalidEpisodeError

from .exploration_parameter_generator import DynamicExplorationParameterGenerator, StaticExplorationParameterGenerator


class Scheduler(object):
    """Scheduler that generates exploration parameters for each episode.

    Args:
        max_ep (int): Maximum number of episodes to be run.
        warmup_ep (int): Episode from which early stopping checking is initiated.
        early_stopping_callback (Callable): Function that returns a boolean indicating whether early stopping should
            be triggered. Defaults to None, in which case no early stopping check will be performed.
        exploration_parameter_generator_cls: Subclass of StaticExplorationParameterGenerator or
            DynamicExplorationParameterGenerator. Defaults to None, which means no exploration outside the algorithm.
        exploration_parameter_generator_config (dict): Configuration for the exploration parameter generator.
            Defaults to None.
        logger (Logger): Used to log important messages.
    """

    def __init__(
        self,
        max_ep: int,
        warmup_ep: int = 0,
        early_stopping_callback: Callable = None,
        exploration_parameter_generator_cls=None,
        exploration_parameter_generator_config: dict = None,
        logger: Logger = DummyLogger()
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
        self._exploration_params = None

        if exploration_parameter_generator_cls is None:
            self._exploration_parameter_generator = None
        elif issubclass(exploration_parameter_generator_cls, StaticExplorationParameterGenerator):
            self._exploration_parameter_generator = exploration_parameter_generator_cls(
                max_ep, **exploration_parameter_generator_config
            )
        elif issubclass(exploration_parameter_generator_cls, DynamicExplorationParameterGenerator):
            self._exploration_parameter_generator = exploration_parameter_generator_cls(
                **exploration_parameter_generator_config
            )

        self._logger = logger

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_ep == self._max_ep:
            raise StopIteration
        if self._current_ep >= self._warmup_ep:
            if self._early_stopping_callback and self._early_stopping_callback(self._performance_history):
                raise StopIteration
        if isinstance(self._exploration_parameter_generator, StaticExplorationParameterGenerator):
            self._exploration_params = self._exploration_parameter_generator.next()
        elif isinstance(self._exploration_parameter_generator, DynamicExplorationParameterGenerator):
            self._exploration_params = self._exploration_parameter_generator.next(self._performance_history)

        return self._exploration_params

    @property
    def current_ep(self):
        return self._current_ep

    def record_performance(self, performance):
        self._performance_history.append(performance)
        self._logger.info(
            f"ep {self._current_ep} - performance: {performance}, exploration_params: {self._exploration_params}"
        )
        self._current_ep += 1
