# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .abs_explorer import AbsExplorer


class LinearExplorer(AbsExplorer):
    """A simple linear exploration scheme."""
    def __init__(self, agent_id_list, total_episodes, epsilon_range_dict, with_cache=True):
        super().__init__(agent_id_list, total_episodes, epsilon_range_dict, with_cache=with_cache)
        self._step_dict = {}
        for agent_id in agent_id_list:
            min_eps, max_eps = self._epsilon_range_dict.get(agent_id, (.0, .0))
            self._step_dict[agent_id] = (max_eps - min_eps) / total_episodes

    def update(self, performance=None):
        for agent_id in self._current_epsilon:
            self._current_epsilon[agent_id] = max(.0, self._current_epsilon[agent_id] - self._step_dict[agent_id])


class TwoPhaseLinearExplorer(AbsExplorer):
    """An exploration scheme that consists of two linear schedules separated by a split point."""
    def __init__(self, agent_id_list, total_episodes, epsilon_range_dict, split_point_dict: dict, with_cache=True):
        super().__init__(agent_id_list, total_episodes, epsilon_range_dict, with_cache=with_cache)
        self._step_dict_p1, self._step_dict_p2, self._num_episodes_p1_dict = {}, {}, {}
        for agent_id in agent_id_list:
            if "_all_" in self._epsilon_range_dict:
                min_eps, max_eps = self._epsilon_range_dict["_all_"]
            else:
                min_eps, max_eps = self._epsilon_range_dict.get(agent_id, (.0, .0))
            eps_range = max_eps - min_eps
            split_point = split_point_dict["_all_"] if "_all_" in split_point_dict else split_point_dict[agent_id]
            num_episodes_p1 = int(total_episodes * split_point[0])
            num_episodes_p2 = total_episodes - num_episodes_p1
            self._step_dict_p1[agent_id] = eps_range * (1 - split_point[1]) / (num_episodes_p1 - 1 + 1e-10)
            self._step_dict_p2[agent_id] = eps_range * split_point[1] / (num_episodes_p2 + 1e-10)
            self._num_episodes_p1_dict[agent_id] = num_episodes_p1

        self._counter = 0

    def update(self, performance=None):
        self._counter += 1
        for agent_id, num_episodes_p1 in self._num_episodes_p1_dict.items():
            step_dict = self._step_dict_p1 if self._counter <= num_episodes_p1 else self._step_dict_p2
            self._current_epsilon[agent_id] = max(.0, self._current_epsilon[agent_id] - step_dict[agent_id])
