# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict


class ExperienceCollectionUtils:
    @staticmethod
    def concat(exp, is_single_source: bool = False, is_single_agent: bool = False) -> dict:
        """Concatenate experiences from multiple sources, by agent ID.

        The experience from each source is expected to be already grouped by agent ID. The result is a single dictionary
        of experiences with keys being agent IDs and values being the concatenation of experiences from all sources
        for each agent ID.

        Args:
            exp: Experiences from one or more sources.
            is_single_source (bool): If True, experiences are from a single (actor) source. Defaults to False.
            is_single_agent (bool): If True, experiences are from a single agent. Defaults to False.

        Returns:
            Concatenated experiences for each agent.
        """
        if is_single_source:
            return exp

        merged = defaultdict(list) if is_single_agent else defaultdict(lambda: defaultdict(list))
        for ex in exp.values():
            if is_single_agent:
                for k, v in ex.items():
                    merged[k].extend[v]
            else:
                for agent_id, e in ex.items():
                    for k, v in e.items():
                        merged[agent_id][k].extend(v)

        return merged

    @staticmethod
    def stack(exp, is_single_source: bool = False, is_single_agent: bool = False) -> dict:
        """Collect each agent's trajectories from multiple sources.

        Args:
            exp: Experiences from one or more sources.
            is_single_source (bool): If True, experiences are from a single (actor) source. Defaults to False.
            is_single_agent (bool): If True, the experiences are from a single agent. Defaults to False.

        Returns:
            A list of trajectories for each agent.
        """
        if is_single_source:
            return [exp] if is_single_agent else {agent_id: [ex] for agent_id, ex in exp.items()}

        if is_single_agent:
            return list(exp.values())

        ret = defaultdict(list)
        for ex in exp.values():
            for agent_id, e in ex.items():
                ret[agent_id].append(e)

        return ret
