# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsExplorer(ABC):
    """Abstract explorer class.

    An explorer is responsible for generating exploration rates.

    Args:
        agent_id_list (list): List of agent ID's.
        total_episodes (int): Total number of episodes in the training phase.
        epsilon_range_dict (dict): A dictionary containing tuples of lower and upper bounds for the generated
            exploration rate for each agent. If the dictionary contains `_all_` as a key, the corresponding
            value will be shared amongst all agents.
        with_cache (bool): If True, incoming performances will be cached.
    """
    def __init__(self, agent_id_list: list, total_episodes: int, epsilon_range_dict: dict, with_cache: bool = True):
        self._total_episodes = total_episodes
        self._epsilon_range_dict = epsilon_range_dict
        self._performance_cache = [] if with_cache else None
        if "_all_" in self._epsilon_range_dict:
            self._current_epsilon = {agent_id: self._epsilon_range_dict["_all_"][1] for agent_id in agent_id_list}
        else:
            self._current_epsilon = {agent_id: self._epsilon_range_dict.get(agent_id, (.0, .0))[1]
                                     for agent_id in agent_id_list}

    # TODO: performance: summary -> total perf (current version), details -> per-agent perf
    @abstractmethod
    def update(self, performance=None):
        """Update exploration rates for each agent.

        Args:
            performance: Performance from the latest episode.
        """
        return NotImplementedError

    @property
    def epsilon_range_dict(self):
        """Exploration rate ranges for each agent."""
        return self._epsilon_range_dict

    @property
    def epsilon(self):
        """Current exploration rates for each agent."""
        return self._current_epsilon

    @epsilon.setter
    def epsilon(self, epsilon_dict: dict):
        self._current_epsilon = epsilon_dict

    def epsilon_range_by_id(self, agent_id):
        return self._epsilon_range_dict[agent_id]
