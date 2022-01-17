# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple


# def get_rollout_finish_msg(ep: int, step_range: Tuple[int, int], exploration_params: dict = None) -> str:
#     """Generate a brief summary message for a finished roll-out"""
#     if exploration_params:
#         exploration_params = {policy_id: params for policy_id, params in exploration_params.items() if params}
#     if exploration_params:
#         return (
#             f"Roll-out finished (episode {ep}, "
#             f"step range: {step_range}, exploration parameters: {exploration_params})"
#         )
#     else:
#         return f"Roll-out finished (episode: {ep}, step range: {step_range})"


def get_rollout_finish_msg(ep: int) -> str:
    """Generate a brief summary message for a finished roll-out"""
    return f"Roll-out finished (episode: {ep})"
