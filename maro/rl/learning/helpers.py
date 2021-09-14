# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union


def get_rollout_finish_msg(ep, step_range, exploration_params=None):
    """Generate a brief summary message for a finished roll-out"""
    if exploration_params:
        return (
            f"Roll-out finished (episode {ep}, "
            f"step range: {step_range}, exploration parameters: {exploration_params})"
        )
    else:
        return f"Roll-out finished (episode: {ep}, step range: {step_range})"


def get_eval_schedule(sch: Union[int, List[int]], num_episodes: int, final: bool = True):
    """Helper function to the policy evaluation schedule.

    Args:
        sch (Union[int, List[int]]): Evaluation schedule. If it is an int, it is treated as the number of episodes
            between two adjacent evaluations. For example, if the total number of episodes is 20 and ``sch`` is 6,
            this will return [6, 12, 18] if ``final`` is False or [6, 12, 18, 20] otherwise. If it is a list, it will
            return a sorted version of the list (with the last episode appended if ``final`` is True).
        num_episodes (int): Total number of learning episodes.
        final (bool): If True, the last episode number will be appended to the returned list to indicate that an
            evaluation is required after the last episode is complete. Defaults to True.

    Returns:
        A list of episodes indicating when to perform policy evaluation.

    """
    if sch is None:
        schedule = []
    elif isinstance(sch, int):
        num_eval_schedule = num_episodes // sch
        schedule = [sch * i for i in range(1, num_eval_schedule + 1)]
    else:
        schedule = sorted(sch)

    # always evaluate after the last episode
    if final and (not schedule or num_episodes != schedule[-1]):
        schedule.append(num_episodes)

    return schedule
