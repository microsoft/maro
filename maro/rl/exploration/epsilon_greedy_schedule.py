# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.utils.exception.rl_toolkit_exception import InvalidEpisodeError


def linear_epsilon_schedule(max_ep: int, start_eps: float, end_eps: float = .0):
    """Linear exploration rate generator for epsilon-greedy exploration.

    Args:
        max_ep (int): Maximum number of episodes to run.
        start_eps (float): The exploration rate for the first episode.
        end_eps (float): The exploration rate for the last episode. Defaults to zero.

    """
    if max_ep <= 0:
        raise InvalidEpisodeError("max_ep must be a positive integer.")
    current_eps = start_eps
    eps_delta = (end_eps - start_eps) / (max_ep - 1)

    for ep in range(max_ep):
        yield current_eps
        current_eps += eps_delta


def two_phase_linear_epsilon_schedule(
    max_ep: int, split_ep: float, start_eps: float, mid_eps: float, end_eps: float = .0
):
    """Exploration schedule comprised of two linear schedules joined together for epsilon-greedy exploration.

    Args:
        max_ep (int): Maximum number of episodes to run.
        split_ep (float): The episode where the switch from the first linear schedule to the second occurs.
        start_eps (float): Exploration rate for the first episode.
        mid_eps (float): The exploration rate where the switch from the first linear schedule to the second occurs.
            In other words, this is the exploration rate where the first linear schedule ends and the second begins.
        end_eps (float): Exploration rate for the last episode. Defaults to zero.

    Returns:
        An iterator over the series of exploration rates from episode 0 to ``max_ep`` - 1.
    """
    if max_ep <= 0:
        raise InvalidEpisodeError("max_ep must be a positive integer.")
    if split_ep <= 0 or split_ep >= max_ep:
        raise ValueError("split_ep must be between 0 and max_ep - 1.")
    current_eps = start_eps
    eps_delta_phase_1 = (mid_eps - start_eps) / split_ep
    eps_delta_phase_2 = (end_eps - mid_eps) / (max_ep - split_ep - 1)
    for ep in range(max_ep):
        yield current_eps
        current_eps += eps_delta_phase_1 if ep < split_ep else eps_delta_phase_2
