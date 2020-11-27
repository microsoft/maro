# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict


def concat_experiences_by_agent(exp_by_source: dict) -> dict:
    """Concatenate experiences from multiple sources, by agent ID.

    The experience from each source is expected to be already grouped by agent ID. The result is a single dictionary
    of experiences with keys being agent IDs and values being the concatenation of experiences from all sources
    for each agent ID.

    Args:
        exp_by_source (dict): Experiences from multiple sources. Each value should consist of experiences grouped by
            agent ID.

    Returns:
        Merged experiences with agent IDs as keys.
    """
    merged = {}
    for exp_by_agent in exp_by_source.values():
        for agent_id, exp in exp_by_agent.items():
            if agent_id not in merged:
                merged[agent_id] = defaultdict(list)
            for k, v in exp.items():
                merged[agent_id][k].extend(v)

    return merged


def merge_experiences_with_trajectory_boundaries(trajectories_by_source) -> dict:
    """Collect each agent's trajectories from multiple sources.

    Args:
        trajectories_by_source (dict): Agent's trajectories from multiple sources.

    Returns:
        A list of trajectories for each agent.
    """
    merged = defaultdict(list)
    for exp_by_agent in trajectories_by_source.values():
        for agent_id, trajectory in exp_by_agent.items():
            merged[agent_id].append(trajectory)

    return merged
