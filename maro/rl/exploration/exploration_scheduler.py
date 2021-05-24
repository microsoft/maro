# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import List, Tuple

from maro.rl.exploration.abs_exploration import AbsExploration


class AbsExplorationScheduler(ABC):
    def __init__(
        self,
        exploration: AbsExploration,
        param_name: str,
        last_ep: int,
        initial_value=None
    ):
        super().__init__()
        self.exploration = exploration
        self.param_name = param_name
        self.last_ep = last_ep
        if initial_value is not None:
            setattr(self.exploration, self.param_name, initial_value)

    def get_value(self):
        return getattr(self.exploration, self.param_name)

    @abstractmethod
    def step(self):
        raise NotImplementedError


class LinearExplorationScheduler(AbsExplorationScheduler):
    """Static exploration parameter generator based on a linear schedule.

    Args:
        max_iter (int): Maximum number of iterations.
        parameter_names (List[str]): List of exploration parameter names.
        start (Union[float, list, tuple, np.ndarray]): Exploration parameter values for the first episode.
            These values must correspond to ``parameter_names``.
        end (Union[float, list, tuple, np.ndarray]): Exploration parameter values rate for the last episode.
            These values must correspond to ``parameter_names``.
    """
    def __init__(
        self,
        exploration: AbsExploration,
        param_name: str,
        last_ep: int,
        final_value: float,
        initial_value: float = None,
    ):
        super().__init__(exploration, param_name, last_ep, initial_value=initial_value)
        self.final_value = final_value
        if self.last_ep > 1:
            self.delta = (self.final_value - getattr(self.exploration, self.param_name)) / (self.last_ep - 1)
        else:
            self.delta = 0

    def step(self):
        if self.get_value() == self.final_value:
            return

        setattr(self.exploration, self.param_name, self.get_value() + self.delta)


class MultiPhaseLinearExplorationScheduler(AbsExplorationScheduler):
    """Exploration parameter generator based on two linear schedules joined together.

    Args:
        max_iter (int): Maximum number of iterations.
        parameter_names (List[str]): List of exploration parameter names.
        split (float): The point where the switch from the first linear schedule to the second occurs.
        start (Union[float, list, tuple, np.ndarray]): Exploration parameter values for the first episode.
            These values must correspond to ``parameter_names``.
        mid (Union[float, list, tuple, np.ndarray]): Exploration parameter values where the switch from the
            first linear schedule to the second occurs. In other words, this is the exploration rate where the first
            linear schedule ends and the second begins. These values must correspond to ``parameter_names``.
        end (Union[float, list, tuple, np.ndarray]): Exploration parameter values for the last episode.
            These values must correspond to ``parameter_names``.

    Returns:
        An iterator over the series of exploration rates from episode 0 to ``max_iter`` - 1.
    """
    def __init__(
        self,
        exploration: AbsExploration,
        param_name: str,
        last_ep: int,
        splits: List[Tuple[int, float]],
        final_value: float,
        initial_value: float = None
    ):
        # validate splits
        splits.append((1, initial_value))
        splits.append((last_ep, final_value))
        splits.sort()
        for (ep, _), (ep2, _) in zip(splits, splits[1:]):
            if ep == ep2:
                raise ValueError("The zeroth element of split points must be unique")

        super().__init__(exploration, param_name, last_ep, initial_value=initial_value)
        self.final_value = final_value
        self._splits = splits
        self._ep = 1
        self._split_index = 1
        self._delta = (self._splits[1][1] - self.get_value()) / (self._splits[1][0] - 1)

    def step(self):
        if self._split_index == len(self._splits):
            return

        setattr(self.exploration, self.param_name, self.get_value() + self._delta)
        self._ep += 1
        if self._ep == self._splits[self._split_index][0]:
            self._split_index += 1
            if self._split_index < len(self._splits):
                self._delta = (
                    (self._splits[self._split_index][1] - self._splits[self._split_index - 1][1]) /
                    (self._splits[self._split_index][0] - self._splits[self._split_index - 1][0])
                )


if __name__ == "__main__":
    from maro.rl.exploration.epsilon_greedy_exploration import EpsilonGreedyExploration
    exploration = EpsilonGreedyExploration(5, epsilon=0.6)
    scheduler = MultiPhaseLinearExplorationScheduler(
        exploration, "epsilon", 20, [(12, 0.25), (6, 0.5), (16, 0.15), (9, 0.4)], .0
    )
    for ep in range(1, scheduler.last_ep + 1):
        print(f"ep = {ep}, value = {exploration.epsilon}")
        scheduler.step()
