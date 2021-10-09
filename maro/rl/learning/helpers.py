# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

def get_rollout_finish_msg(ep, step_range, exploration_params=None):
    """Generate a brief summary message for a finished roll-out"""
    if exploration_params:
        exploration_params = {policy_id: params for policy_id, params in exploration_params.items() if params}
    if exploration_params:
        return (
            f"Roll-out finished (episode {ep}, "
            f"step range: {step_range}, exploration parameters: {exploration_params})"
        )
    else:
        return f"Roll-out finished (episode: {ep}, step range: {step_range})"
