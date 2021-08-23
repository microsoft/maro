# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union


def get_rollout_finish_msg(ep, step_range, exploration_params=None):
    if exploration_params:
        return (
            f"Roll-out finished (episode {ep}, "
            f"step range: {step_range}, exploration parameters: {exploration_params})"
        )
    else:
        return f"Roll-out finished (episode: {ep}, step range: {step_range})"


def get_eval_schedule(sch: Union[int, List[int]], num_episodes: int, final: bool = True):
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
