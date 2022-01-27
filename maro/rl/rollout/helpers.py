# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

def get_rollout_finish_msg(ep: int) -> str:
    """Generate a brief summary message for a finished roll-out

    Args:
        ep (int): Episode number.

    Returns:
        message (str): The message.
    """
    return f"Roll-out finished (episode: {ep})"
